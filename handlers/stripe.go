package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/stripe/stripe-go/v76"
	billingportal "github.com/stripe/stripe-go/v76/billingportal/session"
	checkoutsession "github.com/stripe/stripe-go/v76/checkout/session"
	"github.com/stripe/stripe-go/v76/customer"
	"github.com/stripe/stripe-go/v76/subscription"
	"github.com/stripe/stripe-go/v76/webhook"
)

func initStripe() {
	if stripe.Key == "" {
		stripe.Key = os.Getenv("STRIPE_SECRET_KEY")
	}
}

func cancelExistingSubscriptions(customerID string) {
	iter := subscription.List(&stripe.SubscriptionListParams{
		Customer: stripe.String(customerID),
		Status:   stripe.String("active"),
	})
	for iter.Next() {
		sub := iter.Subscription()
		_, err := subscription.Cancel(sub.ID, nil)
		if err != nil {
			log.Printf("[STRIPE] Failed to cancel subscription %s: %v", sub.ID, err)
		} else {
			log.Printf("[STRIPE] Canceled old subscription %s for customer %s", sub.ID, customerID)
		}
	}
}


func getOrCreateStripeCustomer(userID string, quota *UserQuota) (string, error) {
	initStripe()
	if quota.StripeCustomerID != "" {
		return quota.StripeCustomerID, nil
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		return "", err
	}

	params := &stripe.CustomerParams{
		Email: stripe.String(user.Email),
		Name:  stripe.String(user.Name),
	}
	params.AddMetadata("user_id", userID)

	c, err := customer.New(params)
	if err != nil {
		return "", err
	}

	quota.StripeCustomerID = c.ID
	res := DB.Exec("UPDATE user_quotas SET stripe_customer_id = ?, updated_at = NOW() WHERE user_id = ?", c.ID, userID)
	if res.Error != nil {
		log.Printf("[STRIPE] Failed to save StripeCustomerID: %v (user=%s, c.id=%s)", res.Error, userID, c.ID)
		return "", res.Error
	}
	log.Printf("[STRIPE] Saved StripeCustomerID %s for user %s (rows=%d)", c.ID, userID, res.RowsAffected)
	return c.ID, nil
}

func CreateCheckoutSessionHandler(w http.ResponseWriter, r *http.Request) {
	initStripe()
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Plan         string `json:"plan"`
		BillingCycle string `json:"billing_cycle"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	yearly := req.BillingCycle == "yearly"
	var priceID string
	switch req.Plan {
	case "plus":
		if yearly {
			priceID = os.Getenv("STRIPE_PRICE_PLUS_YEARLY")
		} else {
			priceID = os.Getenv("STRIPE_PRICE_PLUS")
		}
	case "pro":
		if yearly {
			priceID = os.Getenv("STRIPE_PRICE_PRO_YEARLY")
		} else {
			priceID = os.Getenv("STRIPE_PRICE_PRO")
		}
	default:
		http.Error(w, "Invalid plan", http.StatusBadRequest)
		return
	}
	if priceID == "" {
		http.Error(w, fmt.Sprintf("Price ID not configured for plan %s", req.Plan), http.StatusInternalServerError)
		return
	}

	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		http.Error(w, "Quota error", http.StatusInternalServerError)
		return
	}

	customerID, err := getOrCreateStripeCustomer(userID, quota)
	if err != nil {
		log.Printf("[STRIPE] Customer create failed: %v", err)
		http.Error(w, "Stripe customer error", http.StatusInternalServerError)
		return
	}

	cancelExistingSubscriptions(customerID)

	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:3000"
	}

	params := &stripe.CheckoutSessionParams{
		Customer:   stripe.String(customerID),
		Mode:       stripe.String(string(stripe.CheckoutSessionModeSubscription)),
		SuccessURL: stripe.String(baseURL + "/billing?success=1&session_id={CHECKOUT_SESSION_ID}"),
		CancelURL:  stripe.String(baseURL + "/billing?canceled=1"),
		LineItems: []*stripe.CheckoutSessionLineItemParams{
			{
				Price:    stripe.String(priceID),
				Quantity: stripe.Int64(1),
			},
		},
	}
	params.AddMetadata("user_id", userID)
	params.AddMetadata("plan", req.Plan)
	if yearly {
		params.AddMetadata("billing_cycle", "yearly")
	}

	sess, err := checkoutsession.New(params)
	if err != nil {
		log.Printf("[STRIPE] Checkout session failed: %v", err)
		http.Error(w, "Checkout session error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"url": sess.URL, "id": sess.ID})
}

func BuyCreditsHandler(w http.ResponseWriter, r *http.Request) {
	initStripe()
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		AmountUSD float64 `json:"amount_usd"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		http.Error(w, "Quota error", http.StatusInternalServerError)
		return
	}

	var minUSD, maxUSD float64
	switch quota.Plan {
	case "plus":
		minUSD = getEnvPrice("PLUS_CREDIT_MIN", 10)
		maxUSD = getEnvPrice("PLUS_CREDIT_MAX", 500)
	case "pro":
		minUSD = getEnvPrice("PRO_CREDIT_MIN", 10)
		maxUSD = getEnvPrice("PRO_CREDIT_MAX", 5000)
	case "enterprise":
		minUSD = getEnvPrice("ENTERPRISE_CREDIT_MIN", 10)
		maxUSD = getEnvPrice("ENTERPRISE_CREDIT_MAX", 100000)
	case "alpha_unlimited", "unlimited", "limitless":
		minUSD = 10
		maxUSD = 100000
	default:
		http.Error(w, "Credit purchase requires PLUS or PRO plan", http.StatusForbidden)
		return
	}

	if req.AmountUSD < minUSD {
		http.Error(w, fmt.Sprintf("Minimum purchase is $%.0f", minUSD), http.StatusBadRequest)
		return
	}

	remaining := maxUSD - quota.CreditsPurchasedMonth
	if req.AmountUSD > remaining {
		http.Error(w, fmt.Sprintf("Monthly cap exceeded. Remaining this month: $%.2f", remaining), http.StatusBadRequest)
		return
	}

	customerID, err := getOrCreateStripeCustomer(userID, quota)
	if err != nil {
		http.Error(w, "Stripe customer error", http.StatusInternalServerError)
		return
	}

	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:3000"
	}

	amountCents := int64(req.AmountUSD * 100)

	params := &stripe.CheckoutSessionParams{
		Customer:   stripe.String(customerID),
		Mode:       stripe.String(string(stripe.CheckoutSessionModePayment)),
		SuccessURL: stripe.String(baseURL + "/billing?credits=1&session_id={CHECKOUT_SESSION_ID}"),
		CancelURL:  stripe.String(baseURL + "/billing?canceled=1"),
		LineItems: []*stripe.CheckoutSessionLineItemParams{
			{
				PriceData: &stripe.CheckoutSessionLineItemPriceDataParams{
					Currency: stripe.String("usd"),
					ProductData: &stripe.CheckoutSessionLineItemPriceDataProductDataParams{
						Name:        stripe.String("SchemaLabs API Credits"),
						Description: stripe.String(fmt.Sprintf("$%.2f credit top-up, expires end of month", req.AmountUSD)),
					},
					UnitAmount: stripe.Int64(amountCents),
				},
				Quantity: stripe.Int64(1),
			},
		},
	}
	params.AddMetadata("user_id", userID)
	params.AddMetadata("type", "credit_topup")
	params.AddMetadata("amount_usd", fmt.Sprintf("%.2f", req.AmountUSD))

	sess, err := checkoutsession.New(params)
	if err != nil {
		log.Printf("[STRIPE] BuyCredits checkout failed: %v", err)
		http.Error(w, "Checkout error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"url": sess.URL, "id": sess.ID})
}

func CustomerPortalHandler(w http.ResponseWriter, r *http.Request) {
	initStripe()
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

	if quota.StripeCustomerID == "" {
		http.Error(w, "No Stripe customer yet", http.StatusBadRequest)
		return
	}

	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:3000"
	}

	params := &stripe.BillingPortalSessionParams{
		Customer:  stripe.String(quota.StripeCustomerID),
		ReturnURL: stripe.String(baseURL + "/billing"),
	}

	sess, err := billingportal.New(params)
	if err != nil {
		log.Printf("[STRIPE] Portal session failed: %v", err)
		http.Error(w, "Portal session error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"url": sess.URL})
}

type ProcessedWebhookEvent struct {
	EventID   string    `gorm:"primaryKey" json:"event_id"`
	EventType string    `json:"event_type"`
	CreatedAt time.Time `json:"created_at"`
}

func StripeWebhookHandler(w http.ResponseWriter, r *http.Request) {
	initStripe()
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Read error", http.StatusBadRequest)
		return
	}

	secret := os.Getenv("STRIPE_WEBHOOK_SECRET")
	sig := r.Header.Get("Stripe-Signature")

	event, err := webhook.ConstructEventWithOptions(payload, sig, secret, webhook.ConstructEventOptions{IgnoreAPIVersionMismatch: true})
	if err == nil {
		var existing ProcessedWebhookEvent
		if err2 := DB.Where("event_id = ?", event.ID).First(&existing).Error; err2 == nil {
			log.Printf("[STRIPE] duplicate event %s, ack", event.ID)
			w.WriteHeader(http.StatusOK)
			return
		}
	}
	if err != nil {
		log.Printf("[STRIPE_WEBHOOK] Signature verify failed: %v", err)
		http.Error(w, "Signature error", http.StatusBadRequest)
		return
	}

	switch event.Type {
	case "checkout.session.completed":
		var sess stripe.CheckoutSession
		if err := json.Unmarshal(event.Data.Raw, &sess); err != nil {
			log.Printf("[STRIPE_WEBHOOK] Parse failed: %v", err)
			break
		}
		userID := sess.Metadata["user_id"]
		if userID == "" {
			log.Printf("[STRIPE_WEBHOOK] No user_id in session %s", sess.ID)
			break
		}
		quota, err := GetOrCreateQuota(userID)
		if err != nil {
			log.Printf("[STRIPE_WEBHOOK] Quota error: %v", err)
			break
		}
		_ = quota

		if sess.Metadata["type"] == "credit_topup" {
			var amount float64
			fmt.Sscanf(sess.Metadata["amount_usd"], "%f", &amount)
			now := time.Now()
			expire := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())
			res := DB.Exec("UPDATE user_quotas SET credits_balance_usd = credits_balance_usd + ?, credits_purchased_month = credits_purchased_month + ?, credits_expires_at = ?, updated_at = NOW() WHERE user_id = ?", amount, amount, expire, userID)
			log.Printf("[STRIPE_WEBHOOK] Credits +%.2f user=%s err=%v rows=%d", amount, userID, res.Error, res.RowsAffected)
		} else if plan := sess.Metadata["plan"]; plan == "plus" || plan == "pro" {
			var storageMB float64
			switch plan {
			case "plus":
				storageMB = getEnvPrice("PLUS_STORAGE_MB", 5120)
			case "pro":
				storageMB = getEnvPrice("PRO_STORAGE_MB", 51200)
			}
			subID := ""
			if sess.Subscription != nil {
				subID = sess.Subscription.ID
			}
			res := DB.Exec("UPDATE user_quotas SET plan = ?, stripe_subscription_id = ?, storage_limit_mb = ?, updated_at = NOW() WHERE user_id = ?", plan, subID, storageMB, userID)
			if res.Error != nil {
				log.Printf("[STRIPE_WEBHOOK] Plan update failed: %v", res.Error)
			} else {
				log.Printf("[STRIPE_WEBHOOK] Plan upgraded to %s for user %s (rows=%d)", plan, userID, res.RowsAffected)
			}
		}

	case "customer.subscription.deleted":
		var sub stripe.Subscription
		if err := json.Unmarshal(event.Data.Raw, &sub); err != nil {
			log.Printf("[STRIPE_WEBHOOK] sub parse failed: %v", err)
			break
		}
		storageMB := getEnvPrice("FREE_STORAGE_MB", 50)
		res := DB.Exec("UPDATE user_quotas SET plan = 'free', stripe_subscription_id = '', storage_limit_mb = ?, updated_at = NOW() WHERE stripe_subscription_id = ?", storageMB, sub.ID)
		log.Printf("[STRIPE_WEBHOOK] Subscription %s deleted -> free (err=%v rows=%d)", sub.ID, res.Error, res.RowsAffected)

	case "invoice.payment_failed":
		var inv stripe.Invoice
		if err := json.Unmarshal(event.Data.Raw, &inv); err != nil {
			log.Printf("[STRIPE_WEBHOOK] invoice parse failed: %v", err)
			break
		}
		if inv.Customer != nil {
			var quota UserQuota
			if err := DB.Where("stripe_customer_id = ?", inv.Customer.ID).First(&quota).Error; err == nil {
				go sendPaymentFailedEmail(quota.UserID)
				log.Printf("[STRIPE_WEBHOOK] payment_failed email sent for user %s", quota.UserID)
			}
		}

	default:
		log.Printf("[STRIPE_WEBHOOK] Unhandled event: %s", event.Type)
	}

	DB.Create(&ProcessedWebhookEvent{
		EventID:   event.ID,
		EventType: string(event.Type),
		CreatedAt: time.Now(),
	})

	w.WriteHeader(http.StatusOK)
}
