package handlers

import (
	"context"
	"fmt"
	"log"
	"time"
)

func emailThrottle(userID, kind string, ttlSeconds int) bool {
	rc := getRedisClient(); if rc == nil {
		return true
	}
	ctx := context.Background()
	key := "email_sent:" + kind + ":" + userID
	set, err := rc.SetNX(ctx, key, "1", time.Duration(ttlSeconds)*time.Second).Result()
	if err != nil {
		return true
	}
	return set
}


func sendLowBalanceEmail(userID string, balance, threshold float64) {
	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		return
	}
	svc := NewEmailService()
	subject := "SchemaLabs — Low credit balance alert"
	html := fmt.Sprintf(`<h2>Your credit balance is running low</h2>
<p>Hi %s,</p>
<p>Your current add-on credit balance is <b>$%.2f</b>, which is below 20%% of your typical usage.</p>
<p>To avoid interruption of your compute requests, please purchase additional credits.</p>
<p><a href="https://schemalabs.ai/billing">Manage billing →</a></p>
<p>— SchemaLabs Team</p>`, user.Name, balance)
	if err := svc.SendEmail(user.Email, subject, html); err != nil {
		log.Printf("[BILLING_EMAIL] low_balance failed for %s: %v", userID, err)
	}
}

func sendCreditsZeroEmail(userID string) {
	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		return
	}
	svc := NewEmailService()
	subject := "SchemaLabs — Credit balance depleted"
	html := fmt.Sprintf(`<h2>Your credit balance is $0</h2>
<p>Hi %s,</p>
<p>Compute requests are currently blocked because your add-on credit balance has reached zero.</p>
<p>Please top up your credits to resume Schema, Endpoint, Nota, and Fine-tune operations.</p>
<p><a href="https://schemalabs.ai/billing">Buy credits →</a></p>
<p>— SchemaLabs Team</p>`, user.Name)
	if err := svc.SendEmail(user.Email, subject, html); err != nil {
		log.Printf("[BILLING_EMAIL] credits_zero failed for %s: %v", userID, err)
	}
}

func sendTierUnlockEmail(userID string, newTier int, newRate float64) {
	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		return
	}
	svc := NewEmailService()
	subject := fmt.Sprintf("SchemaLabs — Tier %d unlocked", newTier)
	html := fmt.Sprintf(`<h2>You've unlocked Tier %d fine-tuning pricing</h2>
<p>Hi %s,</p>
<p>Thanks to your cumulative compute spend, your fine-tuning rate has dropped to <b>$%.2f per 1M tokens</b>.</p>
<p>This lower rate applies immediately and persists going forward.</p>
<p><a href="https://schemalabs.ai/usage">View usage →</a></p>
<p>— SchemaLabs Team</p>`, newTier, user.Name, newRate)
	if err := svc.SendEmail(user.Email, subject, html); err != nil {
		log.Printf("[BILLING_EMAIL] tier_unlock failed for %s: %v", userID, err)
	}
}

func sendPaymentFailedEmail(userID string) {
	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		return
	}
	svc := NewEmailService()
	subject := "SchemaLabs — Payment failed"
	html := fmt.Sprintf(`<h2>We couldn't process your recent payment</h2>
<p>Hi %s,</p>
<p>Your most recent SchemaLabs subscription charge was declined. Please update your payment method to avoid service interruption.</p>
<p><a href="https://schemalabs.ai/billing">Update payment →</a></p>
<p>— SchemaLabs Team</p>`, user.Name)
	if err := svc.SendEmail(user.Email, subject, html); err != nil {
		log.Printf("[BILLING_EMAIL] payment_failed failed for %s: %v", userID, err)
	}
}
