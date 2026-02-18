package handlers

import (
	"os"
	"strconv"
	"encoding/json"
	"fmt"
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
	if err != nil {
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
	storageLimitMB := getEnvFloat("MAX_TOTAL_STORAGE_MB", 1024.0)
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
	return DB.Save(quota).Error
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
		if quota.ModelsUsed >= quota.ModelsLimit {
			return false, fmt.Sprintf("Model limit reached (%d/%d)", quota.ModelsUsed, quota.ModelsLimit)
		}
		if remaining < CreditPerTrain {
			return false, "Insufficient credits for training"
		}
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

	// Count actual models and storage from DB
	var modelCount int64
	DB.Model(&FineTunedModel{}).Where("user_id = ?", userID).Count(&modelCount)
	quota.ModelsUsed = int(modelCount)

	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
	quota.StorageUsedMB = float64(totalSize) / (1024 * 1024)

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
	DB.Model(&UploadedFile{}).Where("user_id = ? AND (is_merged = ? OR is_merged IS NULL)", userID, false).Count(&datasetCount)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"plan":             quota.Plan,
		"credits_total":    quota.CreditsTotal,
		"credits_used":     quota.CreditsUsed,
		"credits_remaining": quota.CreditsTotal - quota.CreditsUsed,
		"models_limit":     quota.ModelsLimit,
		"models_used":      quota.ModelsUsed,
		"queries_daily":    quota.QueriesDaily,
		"queries_used":     quota.QueriesUsed,
		"storage_limit_mb": quota.StorageLimitMB,
		"storage_used_mb":  quota.StorageUsedMB,
		"reset_date":       quota.ResetDate,
		"days_until_reset": daysUntilReset,
		"datasets_connected": datasetCount,
	})
}
