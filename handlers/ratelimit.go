package handlers

import (
	"encoding/json"
	"context"
	"fmt"
	"net/http"
	"time"
)

type RateLimitKind string

const (
	RateLimitSchema RateLimitKind = "schema"
	RateLimitNota   RateLimitKind = "nota"
)

type RateLimitSpec struct {
	RPM  int
	ITPM int64
	OTPM int64
}

func getRateLimitSpec(plan string, kind RateLimitKind) RateLimitSpec {
	if IsUnlimitedPlan(plan) {
		return RateLimitSpec{RPM: 1_000_000, ITPM: 1_000_000_000, OTPM: 1_000_000_000}
	}
	prefix := planToRateLimitPrefix(plan)
	suffix := "SCHEMA"
	if kind == RateLimitNota {
		suffix = "NOTA"
	}

	if plan == "free" {
		if kind == RateLimitSchema {
			return RateLimitSpec{RPM: 0, ITPM: 0, OTPM: 0}
		}
		return RateLimitSpec{
			RPM:  int(getEnvInt("FREE_RPM_NOTA", 10)),
			ITPM: getEnvInt("FREE_ITPM_NOTA", 100000),
			OTPM: getEnvInt("FREE_OTPM_NOTA", 20000),
		}
	}

	return RateLimitSpec{
		RPM:  int(getEnvInt(fmt.Sprintf("%s_RPM_%s", prefix, suffix), 60)),
		ITPM: getEnvInt(fmt.Sprintf("%s_ITPM_%s", prefix, suffix), 1_000_000),
		OTPM: getEnvInt(fmt.Sprintf("%s_OTPM_%s", prefix, suffix), 100_000),
	}
}

func planToRateLimitPrefix(plan string) string {
	switch plan {
	case "plus":
		return "PLUS"
	case "pro":
		return "PRO"
	default:
		return "FREE"
	}
}

func CheckRateLimit(userID string, kind RateLimitKind, inputTokens, outputTokens int64) (bool, string) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return true, ""
	}
	spec := getRateLimitSpec(quota.Plan, kind)

	if spec.RPM == 0 {
		return false, fmt.Sprintf("%s access not included in %s plan", kind, quota.Plan)
	}

	rc := getRedisClient()
	if rc == nil {
		return true, ""
	}
	ctx := context.Background()

	now := time.Now().Unix() / 60
	rpmKey := fmt.Sprintf("rl:rpm:%s:%s:%d", kind, userID, now)
	itpmKey := fmt.Sprintf("rl:itpm:%s:%s:%d", kind, userID, now)
	otpmKey := fmt.Sprintf("rl:otpm:%s:%s:%d", kind, userID, now)

	rpmCount, _ := rc.Incr(ctx, rpmKey).Result()
	if rpmCount == 1 {
		rc.Expire(ctx, rpmKey, 90*time.Second)
	}
	if int(rpmCount) > spec.RPM {
		return false, fmt.Sprintf("Rate limit exceeded: %d RPM for %s (%s tier)", spec.RPM, kind, quota.Plan)
	}

	if inputTokens > 0 {
		itpmCount, _ := rc.IncrBy(ctx, itpmKey, inputTokens).Result()
		if itpmCount == inputTokens {
			rc.Expire(ctx, itpmKey, 90*time.Second)
		}
		if itpmCount > spec.ITPM {
			return false, fmt.Sprintf("Input token per minute limit exceeded: %d ITPM for %s (%s tier)", spec.ITPM, kind, quota.Plan)
		}
	}

	if outputTokens > 0 {
		otpmCount, _ := rc.IncrBy(ctx, otpmKey, outputTokens).Result()
		if otpmCount == outputTokens {
			rc.Expire(ctx, otpmKey, 90*time.Second)
		}
		if otpmCount > spec.OTPM {
			return false, fmt.Sprintf("Output token per minute limit exceeded: %d OTPM for %s (%s tier)", spec.OTPM, kind, quota.Plan)
		}
	}

	return true, ""
}

func RateLimitResponse(w http.ResponseWriter, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "60")
	w.WriteHeader(http.StatusTooManyRequests)
	fmt.Fprintf(w, `{"error":"%s","status":"rate_limited"}`, reason)
}

func GetRateLimitStatusHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		http.Error(w, "Quota error", http.StatusInternalServerError)
		return
	}

	schemaSpec := getRateLimitSpec(quota.Plan, RateLimitSchema)
	notaSpec := getRateLimitSpec(quota.Plan, RateLimitNota)

	rc := getRedisClient()
	now := time.Now().Unix() / 60

	readCount := func(prefix, kind string) int64 {
		if rc == nil {
			return 0
		}
		key := fmt.Sprintf("rl:%s:%s:%s:%d", prefix, kind, userID, now)
		v, _ := rc.Get(r.Context(), key).Int64()
		return v
	}

	schemaUsage := map[string]interface{}{
		"rpm":          readCount("rpm", string(RateLimitSchema)),
		"itpm":         readCount("itpm", string(RateLimitSchema)),
		"otpm":         readCount("otpm", string(RateLimitSchema)),
		"rpm_limit":    schemaSpec.RPM,
		"itpm_limit":   schemaSpec.ITPM,
		"otpm_limit":   schemaSpec.OTPM,
	}

	notaUsage := map[string]interface{}{
		"rpm":          readCount("rpm", string(RateLimitNota)),
		"itpm":         readCount("itpm", string(RateLimitNota)),
		"otpm":         readCount("otpm", string(RateLimitNota)),
		"rpm_limit":    notaSpec.RPM,
		"itpm_limit":   notaSpec.ITPM,
		"otpm_limit":   notaSpec.OTPM,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"plan":   quota.Plan,
		"schema": schemaUsage,
		"nota":   notaUsage,
	})
}
