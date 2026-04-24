package handlers

import (
"math"
	"os"
	"strconv"
	"strings"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// UserQuota tracks usage and limits per user
type UserQuota struct {
	ID             string    `gorm:"primaryKey" json:"id"`
	UserID         string    `gorm:"uniqueIndex" json:"user_id"`
	Plan           string    `gorm:"default:free" json:"plan"`
	CreditsTotal   float64   `gorm:"default:2.0" json:"credits_total"`
	CreditsUsed    float64   `gorm:"default:0" json:"credits_used"`
	ModelsLimit    int       `gorm:"default:1" json:"models_limit"`
	ModelsUsed     int       `gorm:"default:0" json:"models_used"`
	QueriesDaily   int       `gorm:"default:10" json:"queries_daily"`
	QueriesUsed    int       `gorm:"default:0" json:"queries_used"`
	StorageLimitMB float64   `gorm:"default:50" json:"storage_limit_mb"`
	StorageUsedMB  float64   `gorm:"default:0" json:"storage_used_mb"`
	ResetDate      time.Time `json:"reset_date"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`

	StripeCustomerID     string `gorm:"column:stripe_customer_id" json:"stripe_customer_id,omitempty"`
	StripeSubscriptionID string `gorm:"column:stripe_subscription_id" json:"stripe_subscription_id,omitempty"`

	CreditsBalanceUSD     float64   `gorm:"default:0" json:"credits_balance_usd"`
	CreditsExpiresAt      time.Time `json:"credits_expires_at"`
	CreditsPurchasedMonth float64   `gorm:"default:0" json:"credits_purchased_month"`

	UsageTier              int     `gorm:"default:1" json:"usage_tier"`
EnterpriseTier         int     `gorm:"default:1" json:"enterprise_tier"`
	CumulativeComputeSpend float64 `gorm:"default:0" json:"cumulative_compute_spend"`
	MonthlyTierSpend       float64 `gorm:"default:0" json:"monthly_tier_spend"`

	DailyCreditsUsed float64   `gorm:"default:0" json:"daily_credits_used"`
	LastDailyReset   time.Time `json:"last_daily_reset"`

	SchemaTokensInput       int64 `gorm:"default:0" json:"schema_tokens_input"`
	SchemaTokensOutput      int64 `gorm:"default:0" json:"schema_tokens_output"`
	NotaTokensInput         int64 `gorm:"default:0" json:"nota_tokens_input"`
	NotaTokensOutput        int64 `gorm:"default:0" json:"nota_tokens_output"`
	SyntheticCellsGenerated int64 `gorm:"default:0" json:"synthetic_cells_generated"`
	FineTuneTokens          int64 `gorm:"default:0" json:"fine_tune_tokens"`
	StorageOverageRate      float64 `gorm:"default:0" json:"storage_overage_rate"`
}

// Credit costs
const (
	CreditPerQuery  = 0.02
	CreditPerTrain  = 0.50
	CreditPerUpload = 0.01
)

// GetOrCreateQuota ensures a quota record exists for user

func IsUnlimitedPlan(plan string) bool {
	return plan == "limitless" || plan == "alpha_unlimited" || plan == "unlimited"
}
func GetPlanStorageLimitMB(plan string) float64 {
	getEnvF := func(key string, def float64) float64 {
		if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil && v > 0 {
			return v
		}
		return def
	}
	switch plan {
	case "plus":
		return getEnvF("PLUS_STORAGE_MB", 5120)
	case "pro":
		return getEnvF("PRO_STORAGE_MB", 51200)
	case "alpha", "alpha_unlimited":
		return getEnvF("LIMITLESS_STORAGE_MB", 102400)
	default:
		return getEnvF("FREE_STORAGE_MB", 50)
	}
}

func GetEnterpriseStorageLimitMB(tier int) float64 {
	getEnvF := func(key string, def float64) float64 {
		if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil && v > 0 {
			return v
		}
		return def
	}
	switch tier {
	case 3:
		return getEnvF("ENTERPRISE_T3_STORAGE_MB", 2097152)
	case 2:
		return getEnvF("ENTERPRISE_T2_STORAGE_MB", 512000)
	default:
		return getEnvF("ENTERPRISE_T1_STORAGE_MB", 102400)
	}
}

func GetEnterpriseSeats(tier int) int {
	getEnvI := func(key string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
			return v
		}
		return def
	}
	switch tier {
	case 3:
		return getEnvI("ENTERPRISE_T3_SEATS", 999)
	case 2:
		return getEnvI("ENTERPRISE_T2_SEATS", 25)
	default:
		return getEnvI("ENTERPRISE_T1_SEATS", 10)
	}
}

func GetPlanMaxFileSizeMB(plan string) int {
	getEnvI := func(key string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
			return v
		}
		return def
	}
	if IsUnlimitedPlan(plan) {
		return getEnvI("MAX_FILE_SIZE_MB_UNLIMITED", 500)
	}
	return getEnvI("MAX_FILE_SIZE_MB", 100)
}


func GetOrCreateQuota(userID string) (*UserQuota, error) {
	var quota UserQuota
	err := DB.Raw("SELECT * FROM user_quotas WHERE user_id = ? LIMIT 1", userID).Scan(&quota).Error
if err == nil && quota.ID != "" {
var expected float64
if quota.Plan == "enterprise" {
expected = GetEnterpriseStorageLimitMB(quota.EnterpriseTier)
} else {
expected = GetPlanStorageLimitMB(quota.Plan)
}
if math.Abs(quota.StorageLimitMB-expected) > 0.01 {
DB.Exec("UPDATE user_quotas SET storage_limit_mb = ?, updated_at = NOW() WHERE user_id = ?", expected, userID)
quota.StorageLimitMB = expected
}
return &quota, nil
}
// Quota not found, create new one
{

		now := time.Now()
		nextMonth := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())

			// Get limits from ENV
	getEnvInt := func(key string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
			return v
		}
		return def
	}

	creditsTotal := getEnvPrice("FREE_DAILY_CREDIT_CAP", 2.0)
	modelsLimit := getEnvInt("FREE_MODELS_LIMIT", 1)
	queriesDaily := getEnvInt("FREE_QUERIES_DAILY", 10)
	plan := "free"
	storageLimitMB := GetPlanStorageLimitMB(plan)

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
		return true, ""
	}
	if IsUnlimitedPlan(quota.Plan) {
		return true, ""
	}

	var modelCount int64
	DB.Model(&FineTunedModel{}).Where("user_id = ?", userID).Count(&modelCount)
	quota.ModelsUsed = int(modelCount)

	today := time.Now().Truncate(24 * time.Hour)
	var queryCount int64
	DB.Model(&Message{}).Where("user_id = ? AND role = 'user' AND created_at >= ?", userID, today).Count(&queryCount)
	quota.QueriesUsed = int(queryCount)

	DB.Save(quota)

	switch creditType {
	case "query":
		if quota.QueriesUsed >= quota.QueriesDaily {
			return false, fmt.Sprintf("Daily query limit reached (%d/%d)", quota.QueriesUsed, quota.QueriesDaily)
		}
	case "train":
		if quota.ModelsUsed >= quota.ModelsLimit {
			return false, fmt.Sprintf("Model limit reached (%d/%d). Upgrade your plan to train more models.", quota.ModelsUsed, quota.ModelsLimit)
		}
	}

	return true, ""
}

func CheckCredits(userID string, cost float64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil { return true, "" }
	if IsUnlimitedPlan(quota.Plan) { return true, "" }
	if quota.Plan == "free" {
		now := time.Now()
		if now.Sub(quota.LastDailyReset) >= 24*time.Hour {
			quota.DailyCreditsUsed = 0
			quota.LastDailyReset = now
			DB.Save(quota)
		}
		cap := getEnvPrice("FREE_DAILY_CREDIT_CAP", 2.0)
		remaining := cap - quota.DailyCreditsUsed
		if remaining < cost {
			if remaining < 0 { remaining = 0 }
			return false, fmt.Sprintf("Daily credit cap reached. Remaining: $%.2f, Required: $%.2f. Resets in 24h.", remaining, cost)
		}
		return true, ""
	}
	if quota.CreditsBalanceUSD < cost {
		return false, fmt.Sprintf("Insufficient credits. Balance: $%.2f, Required: $%.2f. Please purchase add-on credits.", quota.CreditsBalanceUSD, cost)
	}
	return true, ""
}

func CheckStorage(userID string, additionalMB float64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil { return true, "" }
	if IsUnlimitedPlan(quota.Plan) { return true, "" }
	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
	usedMB := float64(totalSize) / (1024 * 1024)
	if usedMB+additionalMB <= quota.StorageLimitMB {
		return true, ""
	}
	if quota.Plan == "free" {
		return false, fmt.Sprintf("Upload failed: storage limit reached. You have used %.0fMB of your %.0fMB quota. Please delete some files or upgrade to PLUS.", usedMB, quota.StorageLimitMB)
	}
	overageMB := (usedMB + additionalMB) - quota.StorageLimitMB
	overageRate := quota.StorageOverageRate
	if overageRate <= 0 {
		if quota.Plan == "enterprise" {
			switch quota.EnterpriseTier {
			case 3:
				overageRate = getEnvPrice("ENTERPRISE_T3_STORAGE_OVERAGE_PER_GB", 0.06)
			case 2:
				overageRate = getEnvPrice("ENTERPRISE_T2_STORAGE_OVERAGE_PER_GB", 0.10)
			default:
				overageRate = getEnvPrice("ENTERPRISE_T1_STORAGE_OVERAGE_PER_GB", 0.15)
			}
		} else {
			overageRate = getEnvPrice("STORAGE_OVERAGE_PER_GB", 0.25)
		}
	}
	overageCostUSD := (overageMB / 1024.0) * overageRate
	if quota.CreditsBalanceUSD < overageCostUSD {
		return false, fmt.Sprintf("Storage overage requires $%.4f from credits but balance is $%.2f. Please purchase add-on credits or delete files.", overageCostUSD, quota.CreditsBalanceUSD)
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

func CheckFineTuneConcurrency(userID string) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return true, ""
	}
	if IsUnlimitedPlan(quota.Plan) {
		return true, ""
	}

	var concurrent, queue int
	switch quota.Plan {
	case "free":
		concurrent = int(getEnvInt("FREE_FT_CONCURRENT", 0))
		queue = int(getEnvInt("FREE_FT_QUEUE", 1))
	case "plus":
		concurrent = int(getEnvInt("PLUS_FT_CONCURRENT", 1))
		queue = int(getEnvInt("PLUS_FT_QUEUE", 3))
	case "pro":
		concurrent = int(getEnvInt("PRO_FT_CONCURRENT", 3))
		queue = int(getEnvInt("PRO_FT_QUEUE", 10))
	default:
		return true, ""
	}

	var activeCount int64
	DB.Model(&FineTunedModel{}).Where("user_id = ? AND status IN ?", userID, []string{"training", "queued", "pending"}).Count(&activeCount)

	maxTotal := int64(concurrent + queue)
	if activeCount >= maxTotal {
		return false, fmt.Sprintf("Fine-tuning slot limit reached (%d active/queued of max %d for %s plan). Wait for a job to finish or upgrade your plan.", activeCount, maxTotal, quota.Plan)
	}

	return true, ""
}

func CheckFreeFineTuneLimits(userID string, reqEpochs, reqRows int) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil || quota == nil {
		return true, ""
	}
	if quota.Plan != "free" {
		return true, ""
	}

	maxRows := int(getEnvInt("FREE_FT_MAX_ROWS_PER_JOB", 1000))
	if reqRows > maxRows {
		return false, fmt.Sprintf("Free plan: max %d rows per fine-tune job (got %d). Upgrade to PLUS.", maxRows, reqRows)
	}

	maxEpochs := int(getEnvInt("FREE_FT_MAX_EPOCHS", 200))
	if reqEpochs > maxEpochs {
		return false, fmt.Sprintf("Free plan: max %d epochs per fine-tune job (got %d). Upgrade to PLUS.", maxEpochs, reqEpochs)
	}

	maxJobsPerDay := int(getEnvInt("FREE_FT_MAX_JOBS_PER_DAY", 1))
	since := time.Now().Add(-24 * time.Hour)
	var count int64
	DB.Model(&FineTunedModel{}).Where("user_id = ? AND created_at >= ?", userID, since).Count(&count)
	if int(count) >= maxJobsPerDay {
		return false, fmt.Sprintf("Free plan: max %d fine-tune job per 24h (submitted %d). Upgrade to PLUS.", maxJobsPerDay, count)
	}

	return true, ""
}

func CheckMonthlyTierCeiling(userID string, estSpend float64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil || quota == nil {
		return true, ""
	}
	if IsUnlimitedPlan(quota.Plan) || quota.Plan == "free" || quota.Plan == "enterprise" {
		return true, ""
	}

	var ceiling float64
	switch quota.UsageTier {
	case 1:
		ceiling = getEnvPrice("USAGE_TIER_1_MONTHLY_CEILING", 500)
	case 2:
		ceiling = getEnvPrice("USAGE_TIER_2_MONTHLY_CEILING", 5000)
	default:
		return true, ""
	}

	projected := quota.MonthlyTierSpend + estSpend
	if projected > ceiling {
		return false, fmt.Sprintf("Tier %d monthly spend ceiling reached ($%.2f of $%.2f). Advance to next tier or wait until next month.", quota.UsageTier, quota.MonthlyTierSpend, ceiling)
	}
	return true, ""
}



func StartCreditExpiryCron() {
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		checkAndReset := func() {
			now := time.Now().UTC()
			if now.Day() != 1 || now.Hour() != 0 {
				return
			}
			res := DB.Exec("UPDATE user_quotas SET credits_balance_usd = 0, credits_purchased_month = 0, monthly_tier_spend = 0, updated_at = NOW() WHERE plan IN (?, ?)", "plus", "pro")
			if res.Error != nil {
				log.Printf("[CREDIT_EXPIRY] reset failed: %v", res.Error)
				return
			}
			log.Printf("[CREDIT_EXPIRY] monthly reset done: %d rows affected", res.RowsAffected)
		}
		checkAndReset()
		for range ticker.C {
			checkAndReset()
		}
	}()
}


func getBaseModelID() string {
	if v := os.Getenv("BASE_MODEL"); v != "" {
		return v
	}
	return "schema-v1"
}

func getBaseModelVersion() string {
	id := getBaseModelID()
	if strings.Contains(id, "v1") {
		return "v1"
	}
	if strings.Contains(id, "v0") {
		return "v0"
	}
	return "v1"
}
