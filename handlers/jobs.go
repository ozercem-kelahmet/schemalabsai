package handlers

import (
	"log"
	"time"
)

func StartBillingJobs() {
	go dailyFreeCreditResetLoop()
	go monthlyCreditExpireLoop()
}

func dailyFreeCreditResetLoop() {
	time.Sleep(30 * time.Second)
	for {
		resetFreeCredits()
		time.Sleep(1 * time.Hour)
	}
}

func resetFreeCredits() {
	if DB == nil {
		return
	}
	cutoff := time.Now().Add(-24 * time.Hour)
	result := DB.Model(&UserQuota{}).
		Where("plan = ? AND last_daily_reset < ?", "free", cutoff).
		Updates(map[string]interface{}{
			"daily_credits_used": 0,
			"last_daily_reset":   time.Now(),
		})
	if result.RowsAffected > 0 {
		log.Printf("[BILLING_JOBS] Reset daily credits for %d free users", result.RowsAffected)
	}
}

func monthlyCreditExpireLoop() {
	time.Sleep(60 * time.Second)
	for {
		expireMonthlyCredits()
		time.Sleep(1 * time.Hour)
	}
}

func expireMonthlyCredits() {
	if DB == nil {
		return
	}
	now := time.Now()
	result := DB.Model(&UserQuota{}).
		Where("credits_expires_at IS NOT NULL AND credits_expires_at < ? AND credits_balance_usd > 0", now).
		Updates(map[string]interface{}{
			"credits_balance_usd":     0,
			"credits_purchased_month": 0,
			"monthly_tier_spend":      0,
			"credits_expires_at":      time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location()),
		})
	if result.RowsAffected > 0 {
		log.Printf("[BILLING_JOBS] Expired monthly credits for %d users", result.RowsAffected)
	}
}
