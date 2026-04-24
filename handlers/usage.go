package handlers

import (
	"net/http"
	"time"
	"os"
	"strconv"
	"encoding/json"
	"fmt"
)

func getEnvPrice(key string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil && v > 0 {
		return v
	}
	return def
}

func TrackSchemaCall(userID string, rows, cols, outputRows int, version string, isEndpoint bool) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	inputTokens := int64(rows * cols)
	outputTokens := int64(outputRows)

	if ok, reason := CheckRateLimit(userID, RateLimitSchema, inputTokens, outputTokens); !ok {
		return fmt.Errorf("%s", reason)
	}

	var inPrice, outPrice float64
	if isEndpoint {
		if version == "v1" {
			inPrice = getEnvPrice("ENDPOINT_V1_INPUT", 3.0)
			outPrice = getEnvPrice("ENDPOINT_V1_OUTPUT", 7.0)
		} else {
			inPrice = getEnvPrice("ENDPOINT_V0_INPUT", 2.0)
			outPrice = getEnvPrice("ENDPOINT_V0_OUTPUT", 4.0)
		}
	} else {
		if version == "v1" {
			inPrice = getEnvPrice("SCHEMA_V1_INPUT", 2.0)
			outPrice = getEnvPrice("SCHEMA_V1_OUTPUT", 6.0)
		} else {
			inPrice = getEnvPrice("SCHEMA_V0_INPUT", 1.0)
			outPrice = getEnvPrice("SCHEMA_V0_OUTPUT", 3.0)
		}
	}

	cost := (float64(inputTokens)/1_000_000.0)*inPrice + (float64(outputTokens)/1_000_000.0)*outPrice

	if err := DeductCredits(userID, cost); err != nil {
		return err
	}

	if err := DB.Exec("UPDATE user_quotas SET schema_tokens_input = schema_tokens_input + ?, schema_tokens_output = schema_tokens_output + ?, cumulative_compute_spend = cumulative_compute_spend + ?, monthly_tier_spend = monthly_tier_spend + ?, updated_at = NOW() WHERE user_id = ?", inputTokens, outputTokens, cost, cost, userID).Error; err != nil {
		return err
	}
	return UpdateUsageTier(userID)
}

func GetUsageLogsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	var logs []UsageLog
	query := DB.Where("user_id = ?", userID).Order("created_at DESC")
	if lim := r.URL.Query().Get("limit"); lim != "" {
		if n, err := strconv.Atoi(lim); err == nil && n > 0 && n <= 1000 {
			query = query.Limit(n)
		}
	} else {
		query = query.Limit(500)
	}
	query.Find(&logs)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"logs": logs})
}

func TrackNotaCall(userID string, inputTokens, outputTokens int64) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	if ok, reason := CheckRateLimit(userID, RateLimitNota, inputTokens, outputTokens); !ok {
		return fmt.Errorf("%s", reason)
	}

	inPrice := getEnvPrice("NOTA_INPUT", 1.0)
	outPrice := getEnvPrice("NOTA_OUTPUT", 4.0)
	cost := (float64(inputTokens)/1_000_000.0)*inPrice + (float64(outputTokens)/1_000_000.0)*outPrice

	if err := DeductCredits(userID, cost); err != nil {
		return err
	}

	if err := DB.Exec("UPDATE user_quotas SET nota_tokens_input = nota_tokens_input + ?, nota_tokens_output = nota_tokens_output + ?, cumulative_compute_spend = cumulative_compute_spend + ?, monthly_tier_spend = monthly_tier_spend + ?, updated_at = NOW() WHERE user_id = ?", inputTokens, outputTokens, cost, cost, userID).Error; err != nil {
return err
}
return UpdateUsageTier(userID)
}

func TrackFineTuneJob(userID string, rows, epochs int) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	tokens := int64(rows * epochs)

	var rate float64
	switch quota.UsageTier {
	case 3:
		rate = getEnvPrice("FT_TIER_3_RATE", 3.0)
	case 2:
		rate = getEnvPrice("FT_TIER_2_RATE", 4.5)
	default:
		rate = getEnvPrice("FT_TIER_1_RATE", 6.0)
	}

	cost := (float64(tokens) / 1_000_000.0) * rate

	if err := DeductCredits(userID, cost); err != nil {
		return err
	}

	if err := DB.Exec("UPDATE user_quotas SET fine_tune_tokens = fine_tune_tokens + ?, cumulative_compute_spend = cumulative_compute_spend + ?, monthly_tier_spend = monthly_tier_spend + ?, updated_at = NOW() WHERE user_id = ?", tokens, cost, cost, userID).Error; err != nil {
		return err
	}
	return UpdateUsageTier(userID)
}

func TrackSyntheticGen(userID string, cellCount int64) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	price := getEnvPrice("SYNTHETIC_PER_1M_CELLS", 1.5)
	cost := (float64(cellCount) / 1_000_000.0) * price

	if err := DeductCredits(userID, cost); err != nil {
		return err
	}

	if err := DB.Exec("UPDATE user_quotas SET synthetic_cells_generated = synthetic_cells_generated + ?, cumulative_compute_spend = cumulative_compute_spend + ?, monthly_tier_spend = monthly_tier_spend + ?, updated_at = NOW() WHERE user_id = ?", cellCount, cost, cost, userID).Error; err != nil {
		return err
	}
	return UpdateUsageTier(userID)
}

func UpdateUsageTier(userID string) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}

	tier2 := getEnvPrice("USAGE_TIER_2_THRESHOLD", 500.0)
	tier3 := getEnvPrice("USAGE_TIER_3_THRESHOLD", 5000.0)

	newTier := 1
	if quota.CumulativeComputeSpend >= tier3 {
		newTier = 3
	} else if quota.CumulativeComputeSpend >= tier2 {
		newTier = 2
	}

	if newTier > quota.UsageTier {
		oldTier := quota.UsageTier
		if err := DB.Exec("UPDATE user_quotas SET usage_tier = ?, updated_at = NOW() WHERE user_id = ?", newTier, userID).Error; err != nil {
			return err
		}
		if oldTier < newTier {
			var newRate float64
			switch newTier {
			case 3:
				newRate = getEnvPrice("FT_TIER_3_RATE", 3.0)
			case 2:
				newRate = getEnvPrice("FT_TIER_2_RATE", 4.5)
			default:
				newRate = getEnvPrice("FT_TIER_1_RATE", 6.0)
			}
			go sendTierUnlockEmail(userID, newTier, newRate)
		}
		return nil
	}
	return nil
}


func DeductCredits(userID string, amountUSD float64) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	if quota.Plan == "free" {
		now := time.Now()
		currentDaily := quota.DailyCreditsUsed
		if now.Sub(quota.LastDailyReset) >= 24*time.Hour {
			currentDaily = 0
			if err := DB.Exec("UPDATE user_quotas SET daily_credits_used = 0, last_daily_reset = ? WHERE user_id = ?", now, userID).Error; err != nil {
				return err
			}
		}
		cap := getEnvPrice("FREE_DAILY_CREDIT_CAP", 2.0)
		if currentDaily+amountUSD > cap {
			return fmt.Errorf("daily credit cap reached ($%.2f/$%.2f)", currentDaily, cap)
		}
		return DB.Exec("UPDATE user_quotas SET daily_credits_used = daily_credits_used + ?, updated_at = NOW() WHERE user_id = ?", amountUSD, userID).Error
	}

	if quota.CreditsBalanceUSD < amountUSD {
		go sendCreditsZeroEmail(userID)
		return fmt.Errorf("insufficient credit balance ($%.2f < $%.2f)", quota.CreditsBalanceUSD, amountUSD)
	}
	balanceBefore := quota.CreditsBalanceUSD
	if err := DB.Exec("UPDATE user_quotas SET credits_balance_usd = credits_balance_usd - ?, updated_at = NOW() WHERE user_id = ? AND credits_balance_usd >= ?", amountUSD, userID, amountUSD).Error; err != nil {
		return err
	}
	newBalance := balanceBefore - amountUSD
	threshold := quota.CreditsPurchasedMonth * 0.20
	if threshold > 0 && balanceBefore >= threshold && newBalance < threshold {
		go sendLowBalanceEmail(userID, newBalance, threshold)
	}
	if newBalance <= 0 && balanceBefore > 0 {
		go sendCreditsZeroEmail(userID)
	}
	return nil
}

func TrackFrontierCall(userID string, inputTokens, outputTokens int64, model string) error {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return err
	}
	if IsUnlimitedPlan(quota.Plan) {
		return nil
	}

	if ok, reason := CheckRateLimit(userID, RateLimitNota, inputTokens, outputTokens); !ok {
		return fmt.Errorf("%s", reason)
	}

	baseIn, baseOut := GetFrontierRate(model)

	var markup float64
	if quota.Plan == "enterprise" {
		switch quota.EnterpriseTier {
		case 3:
			markup = getEnvPrice("FRONTIER_MARKUP_T3", 1.10)
		case 2:
			markup = getEnvPrice("FRONTIER_MARKUP_T2", 1.15)
		default:
			markup = getEnvPrice("FRONTIER_MARKUP_T1", 1.20)
		}
	} else {
		markup = getEnvPrice("FRONTIER_MARKUP_T1", 1.20)
	}

	cost := (float64(inputTokens)/1_000_000.0)*baseIn*markup + (float64(outputTokens)/1_000_000.0)*baseOut*markup

	if err := DeductCredits(userID, cost); err != nil {
		return err
	}

	if err := DB.Exec("UPDATE user_quotas SET nota_tokens_input = nota_tokens_input + ?, nota_tokens_output = nota_tokens_output + ?, cumulative_compute_spend = cumulative_compute_spend + ?, monthly_tier_spend = monthly_tier_spend + ?, updated_at = NOW() WHERE user_id = ?", inputTokens, outputTokens, cost, cost, userID).Error; err != nil {
return err
}
return UpdateUsageTier(userID)
}
