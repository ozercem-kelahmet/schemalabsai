package handlers

import (
"math"
	"os"
	"strconv"
	"encoding/json"
	"fmt"
	"strings"
	"net/http"
	"time"
)

// UserQuota tracks usage and limits per user
type UserQuota struct {
	ID             string    `gorm:"primaryKey" json:"id"`
	UserID         string    `gorm:"uniqueIndex" json:"user_id"`
	Plan           string    `gorm:"default:alpha" json:"plan"`
	CreditsTotal   float64   `gorm:"default:5.0" json:"credits_total"`
	CreditsUsed    float64   `gorm:"default:0" json:"credits_used"`
	ModelsLimit    int       `gorm:"default:5" json:"models_limit"`
	ModelsUsed     int       `gorm:"default:0" json:"models_used"`
	QueriesDaily   int       `gorm:"default:10" json:"queries_daily"`
	QueriesUsed    int       `gorm:"default:0" json:"queries_used"`
	StorageLimitMB float64   `gorm:"default:1024" json:"storage_limit_mb"`
	StorageUsedMB  float64   `gorm:"default:0" json:"storage_used_mb"`
	ResetDate      time.Time `json:"reset_date"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// Credit costs
const (
	CreditPerQuery  = 0.02
	CreditPerTrain  = 0.50
	CreditPerUpload = 0.01
)

// GetOrCreateQuota ensures a quota record exists for user
func GetOrCreateQuota(userID string) (*UserQuota, error) {
	var quota UserQuota
	err := DB.Raw("SELECT * FROM user_quotas WHERE user_id = ? LIMIT 1", userID).Scan(&quota).Error
if err == nil && quota.ID != "" {
return &quota, nil
}
// Quota not found, create new one
{
		// Check if existing user (created before 2026-02-04)
		var user User
		DB.Where("id = ?", userID).First(&user)
		cutoff := time.Date(2026, 2, 4, 0, 0, 0, 0, time.UTC)
		isExisting := user.CreatedAt.Before(cutoff)

		now := time.Now()
		nextMonth := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())

			// Get limits from ENV
	getEnvFloat := func(key string, def float64) float64 {
		if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil && v > 0 {
			return v
		}
		return def
	}
	getEnvInt := func(key string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
			return v
		}
		return def
	}

	creditsTotal := getEnvFloat("ALPHA_CREDITS_TOTAL", 5.0)
	modelsLimit := getEnvInt("ALPHA_MODELS_LIMIT", 5)
	queriesDaily := getEnvInt("ALPHA_QUERIES_DAILY", 10)
	storageLimitMB := getEnvFloat("ALPHA_STORAGE_LIMIT_MB", 100.0)
	plan := "alpha"
	
	if isExisting {
		creditsTotal = getEnvFloat("UNLIMITED_CREDITS_TOTAL", 9999.0)
		modelsLimit = getEnvInt("UNLIMITED_MODELS_LIMIT", 9999)
		queriesDaily = getEnvInt("UNLIMITED_QUERIES_DAILY", 9999)
		storageLimitMB = getEnvFloat("MAX_TOTAL_STORAGE_MB_UNLIMITED", 10240.0)
		plan = "alpha_unlimited"
	}

	quota = UserQuota{
		ID:             fmt.Sprintf("quota-%s", userID[:8]),
		UserID:         userID,
		Plan:           plan,
		CreditsTotal:   creditsTotal,
		CreditsUsed:    0,
		ModelsLimit:    modelsLimit,
		ModelsUsed:     0,
		QueriesDaily:   queriesDaily,
		QueriesUsed:    0,
		StorageLimitMB: storageLimitMB,
			StorageUsedMB:  0,
			ResetDate:      nextMonth,
			CreatedAt:      now,
			UpdatedAt:      now,
		}
		DB.Create(&quota)
	}

	// Check if reset needed (monthly)
	if time.Now().After(quota.ResetDate) {
		quota.CreditsUsed = 0
		quota.QueriesUsed = 0
		now := time.Now()
		quota.ResetDate = time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())
		quota.UpdatedAt = now
		DB.Save(&quota)
	}

	return &quota, nil
}

// UseCredit deducts credits and increments counters
func UseCredit(userID string, creditType string) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}

	var cost float64
	switch creditType {
	case "query":
		cost = CreditPerQuery
		quota.QueriesUsed++
	case "train":
		cost = CreditPerTrain
		quota.ModelsUsed++
	case "upload":
		cost = CreditPerUpload
	default:
		cost = 0.01
	}

	quota.CreditsUsed += cost
	quota.UpdatedAt = time.Now()
	if err := DB.Save(quota).Error; err != nil {
		return err
	}
	go notifyThresholds(userID, quota)
	return nil
}

// CheckQuota returns true if user has remaining quota
func CheckQuota(userID string, creditType string) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return true, "" // Allow on error
	}

	// Get real counts from DB
	var modelCount int64
	if err := DB.Model(&FineTunedModel{}).Where("user_id = ?", userID).Count(&modelCount).Error; err != nil {
	}
	quota.ModelsUsed = int(modelCount)

	var totalCreditsUsed float64
	if err := DB.Model(&UsageLog{}).Where("user_id = ?", userID).Select("COALESCE(SUM(credits_used), 0)").Scan(&totalCreditsUsed).Error; err != nil {
	}
	quota.CreditsUsed = totalCreditsUsed

	today := time.Now().Truncate(24 * time.Hour)
	var queryCount int64
	DB.Model(&Message{}).Where("user_id = ? AND role = 'user' AND created_at >= ?", userID, today).Count(&queryCount)
	quota.QueriesUsed = int(queryCount)

	DB.Save(quota)

	remaining := quota.CreditsTotal - quota.CreditsUsed

	switch creditType {
	case "query":
		if quota.QueriesUsed >= quota.QueriesDaily {
			return false, "Daily query limit reached (10/day)"
		}
		if remaining < CreditPerQuery {
			return false, "Insufficient credits"
		}
	case "train":
		var errors []string
		if quota.ModelsUsed >= quota.ModelsLimit {
			errors = append(errors, fmt.Sprintf("Model limit reached (%d/%d)", quota.ModelsUsed, quota.ModelsLimit))
		}
		if remaining < CreditPerTrain {
			dr := remaining; if dr < 0 { dr = 0 }; errors = append(errors, fmt.Sprintf("Insufficient credits. Remaining: %.2f, Required: %.2f", dr, CreditPerTrain))
		}
		if len(errors) > 0 {
			return false, strings.Join(errors, " | ")
		}
	}

	return true, ""
}

// CheckCredits returns false if user has insufficient credits
func CheckCredits(userID string, cost float64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil { return true, "" }
	remaining := quota.CreditsTotal - quota.CreditsUsed
	if remaining < cost {
		dr := remaining; if dr < 0 { dr = 0 }; return false, fmt.Sprintf("Insufficient credits. Remaining: %.2f, Required: %.2f", dr, cost)
	}
	return true, ""
}

// CheckStorage returns false if user has insufficient storage
func CheckStorage(userID string, additionalMB float64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil { return true, "" }
	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
usedMB := float64(totalSize) / (1024 * 1024)
	if usedMB + additionalMB > quota.StorageLimitMB {
		return false, fmt.Sprintf("Upload failed: storage limit reached. You have used %.0fMB of your %.0fMB quota. Please delete some files.", usedMB, quota.StorageLimitMB)
	}
	return true, ""
}

// QuotaHandler returns user's quota info
func QuotaHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		http.Error(w, "Failed to get quota", http.StatusInternalServerError)
		return
	}

// Sync credits from usage_logs
var totalCreditsFromLogs float64
DB.Model(&UsageLog{}).Where("user_id = ?", userID).Select("COALESCE(SUM(credits_used), 0)").Scan(&totalCreditsFromLogs)
if totalCreditsFromLogs > quota.CreditsUsed {
quota.CreditsUsed = totalCreditsFromLogs
}
	// Count actual models and storage from DB
	var modelCount int64
	DB.Model(&FineTunedModel{}).Where("user_id = ?", userID).Count(&modelCount)
	quota.ModelsUsed = int(modelCount)

	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
// Also count connection data size from table_details
var connFiles []Connection
DB.Where("user_id = ?", userID).Find(&connFiles)
var connSizeMB float64
for _, c := range connFiles {
if c.CachedTables != "" && c.CachedTables != "null" && c.CachedTables != "[]" {
var cached struct{ TableDetails []struct{ Rows int `json:"rows"`; Columns int `json:"columns"` } `json:"table_details"` }
if json.Unmarshal([]byte(c.CachedTables), &cached) == nil {
for _, t := range cached.TableDetails {
cols := t.Columns; if cols < 10 { cols = 10 }
connSizeMB += float64(t.Rows * cols * 20) / (1024 * 1024)
}
}
}
}
quota.StorageUsedMB = float64(totalSize) / (1024*1024) + connSizeMB

	// Count today's queries
	today := time.Now().Truncate(24 * time.Hour)
	var queryCount int64
	DB.Model(&Message{}).Where("user_id = ? AND role = 'user' AND created_at >= ?", userID, today).Count(&queryCount)
	quota.QueriesUsed = int(queryCount)

	DB.Save(quota)

	// Days until reset
	daysUntilReset := int(time.Until(quota.ResetDate).Hours() / 24)
	if daysUntilReset < 0 {
		daysUntilReset = 0
	}

	// Count connected datasets
	var datasetCount int64
var connCount int64
DB.Model(&Connection{}).Where("user_id = ?", userID).Count(&connCount)
	DB.Model(&UploadedFile{}).Where("user_id = ? AND (is_merged = ? OR is_merged IS NULL)", userID, false).Count(&datasetCount)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"plan":             quota.Plan,
		"credits_total":    quota.CreditsTotal,
		"credits_used":     math.Min(quota.CreditsUsed, quota.CreditsTotal),
		"credits_remaining": math.Max(0, quota.CreditsTotal - quota.CreditsUsed),
		"models_limit":     quota.ModelsLimit,
		"models_used":      quota.ModelsUsed,
		"queries_daily":    quota.QueriesDaily,
		"queries_used":     quota.QueriesUsed,
		"storage_limit_mb": quota.StorageLimitMB,
		"storage_used_mb":  quota.StorageUsedMB,
		"reset_date":       quota.ResetDate,
		"days_until_reset": daysUntilReset,
		"datasets_connected": datasetCount + connCount,
	})
}

// notifyThresholds checks and sends warning emails if thresholds crossed
func notifyThresholds(userID string, quota *UserQuota) {
	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil || user.Email == "" {
		return
	}

	emailSvc := NewEmailService()
	remaining := quota.CreditsTotal - quota.CreditsUsed
	creditPct := remaining / quota.CreditsTotal * 100

	// Low credit: notify at 20%
	if creditPct <= 20 && creditPct > 0 {
		var count int64
		DB.Model(&UsageLog{}).Where("user_id = ? AND event_type = 'low_credit_warning' AND created_at > ?", userID, time.Now().AddDate(0, 0, -3)).Count(&count)
		if count == 0 {
			go emailSvc.SendLowCreditWarning(user.Email, user.Name, remaining, quota.CreditsTotal)
			DB.Create(&UsageLog{ID: fmt.Sprintf("warn-%d", time.Now().UnixNano()), UserID: userID, EventType: "low_credit_warning", CreditsUsed: 0, CreatedAt: time.Now()})
		}
	}

	// Storage warning: notify at 80%
	storagePct := quota.StorageUsedMB / quota.StorageLimitMB * 100
	if storagePct >= 80 {
		var count int64
		DB.Model(&UsageLog{}).Where("user_id = ? AND event_type = 'storage_warning' AND created_at > ?", userID, time.Now().AddDate(0, 0, -3)).Count(&count)
		if count == 0 {
			go emailSvc.SendStorageWarning(user.Email, user.Name, quota.StorageUsedMB, quota.StorageLimitMB)
			DB.Create(&UsageLog{ID: fmt.Sprintf("warn-%d", time.Now().UnixNano()), UserID: userID, EventType: "storage_warning", CreditsUsed: 0, CreatedAt: time.Now()})
		}
	}
}
