package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type BillingSummary struct {
	Plan                    string  `json:"plan"`
	CreditsRemaining        float64 `json:"creditsRemaining"`
	CreditsUsed             float64 `json:"creditsUsed"`
	CreditsPurchased        float64 `json:"creditsPurchased"`
	DailyCreditCap          float64 `json:"dailyCreditCap,omitempty"`
	DailyCreditsUsed        float64 `json:"dailyCreditsUsed,omitempty"`
	StorageUsedMb           float64 `json:"storageUsedMb"`
	StorageLimitMb          float64 `json:"storageLimitMb"`
	NextResetDate           string  `json:"nextResetDate"`
	UsageTier               int     `json:"usageTier"`
	CumulativeSpend         float64 `json:"cumulativeSpend"`
	MonthlyTierSpend        float64 `json:"monthlyTierSpend"`
	SchemaTokensInput       int64   `json:"schemaTokensInput"`
	SchemaTokensOutput      int64   `json:"schemaTokensOutput"`
	NotaTokensInput         int64   `json:"notaTokensInput"`
	NotaTokensOutput        int64   `json:"notaTokensOutput"`
	SyntheticCellsGenerated int64   `json:"syntheticCellsGenerated"`
	FineTuneTokens          int64   `json:"fineTuneTokens"`
	NextTierThreshold       float64 `json:"nextTierThreshold"`
	MonthlyTierCeiling      float64 `json:"monthlyTierCeiling"`
	FTSlotsUsed             int64   `json:"ftSlotsUsed"`
	FTConcurrentLimit       int     `json:"ftConcurrentLimit"`
	FTQueueLimit            int     `json:"ftQueueLimit"`
}

func GetBillingSummaryHandler(w http.ResponseWriter, r *http.Request) {
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

	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
	storageUsedMb := float64(totalSize) / (1024 * 1024)

	var nextTier, monthCeiling float64
	switch quota.UsageTier {
	case 1:
		nextTier = getEnvPrice("USAGE_TIER_2_THRESHOLD", 500)
		monthCeiling = getEnvPrice("USAGE_TIER_1_MONTHLY_CEILING", 500)
	case 2:
		nextTier = getEnvPrice("USAGE_TIER_3_THRESHOLD", 5000)
		monthCeiling = getEnvPrice("USAGE_TIER_2_MONTHLY_CEILING", 5000)
	default:
		nextTier = 0
		monthCeiling = 0
	}

	var ftSlots int64
	DB.Model(&FineTunedModel{}).Where("user_id = ? AND status IN ?", userID, []string{"training", "queued", "pending"}).Count(&ftSlots)

	var ftConc, ftQueue int
	switch quota.Plan {
	case "free":
		ftConc = int(getEnvInt("FREE_FT_CONCURRENT", 0))
		ftQueue = int(getEnvInt("FREE_FT_QUEUE", 1))
	case "plus":
		ftConc = int(getEnvInt("PLUS_FT_CONCURRENT", 1))
		ftQueue = int(getEnvInt("PLUS_FT_QUEUE", 3))
	case "pro":
		ftConc = int(getEnvInt("PRO_FT_CONCURRENT", 3))
		ftQueue = int(getEnvInt("PRO_FT_QUEUE", 10))
	}

	summary := BillingSummary{
		FTSlotsUsed:             ftSlots,
		FTConcurrentLimit:       ftConc,
		FTQueueLimit:            ftQueue,
		Plan:                    quota.Plan,
		StorageUsedMb:           storageUsedMb,
		StorageLimitMb:          quota.StorageLimitMB,
		UsageTier:               quota.UsageTier,
		CumulativeSpend:         quota.CumulativeComputeSpend,
		CreditsPurchased:        quota.CreditsPurchasedMonth,
		MonthlyTierSpend:        quota.MonthlyTierSpend,
		SchemaTokensInput:       quota.SchemaTokensInput,
		SchemaTokensOutput:      quota.SchemaTokensOutput,
		NotaTokensInput:         quota.NotaTokensInput,
		NotaTokensOutput:        quota.NotaTokensOutput,
		SyntheticCellsGenerated: quota.SyntheticCellsGenerated,
		FineTuneTokens:          quota.FineTuneTokens,
		NextTierThreshold:       nextTier,
		MonthlyTierCeiling:      monthCeiling,
	}

	if quota.Plan == "free" {
		cap := getEnvPrice("FREE_DAILY_CREDIT_CAP", 2.0)
		if time.Since(quota.LastDailyReset) >= 24*time.Hour {
			quota.DailyCreditsUsed = 0
		}
		summary.DailyCreditCap = cap
		summary.DailyCreditsUsed = quota.DailyCreditsUsed
		summary.CreditsRemaining = cap - quota.DailyCreditsUsed
		if summary.CreditsRemaining < 0 {
			summary.CreditsRemaining = 0
		}
		summary.CreditsUsed = quota.DailyCreditsUsed
		nextReset := quota.LastDailyReset.Add(24 * time.Hour)
		if nextReset.Before(time.Now()) {
			nextReset = time.Now().Add(24 * time.Hour)
		}
		summary.NextResetDate = nextReset.Format(time.RFC3339)
	} else {
		summary.CreditsRemaining = quota.CreditsBalanceUSD
		summary.CreditsUsed = quota.MonthlyTierSpend
		if !quota.CreditsExpiresAt.IsZero() {
			summary.NextResetDate = quota.CreditsExpiresAt.Format(time.RFC3339)
		} else {
			now := time.Now()
			nextMonth := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location())
			summary.NextResetDate = nextMonth.Format(time.RFC3339)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(summary)
}

func ensureStripeCustomer(userID string, quota *UserQuota) (string, error) {
	if quota.StripeCustomerID != "" {
		return quota.StripeCustomerID, nil
	}
	return "", nil
}

func ContactSalesHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Email string `json:"email"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	req.Email = strings.ToLower(strings.TrimSpace(req.Email))
	if req.Email == "" {
		http.Error(w, "Email required", http.StatusBadRequest)
		return
	}
	if len(req.Email) > 254 || !isValidEmail(req.Email) {
		http.Error(w, "Invalid email format", http.StatusBadRequest)
		return
	}

	var user User
	DB.Where("id = ?", userID).First(&user)

	quota, _ := GetOrCreateQuota(userID)
	currentPlan := "unknown"
	if quota != nil {
		currentPlan = quota.Plan
	}

	svc := NewEmailService()
	salesRecipients := []string{}
	if env := os.Getenv("SALES_EMAIL"); env != "" {
		for _, r := range strings.Split(env, ",") {
			if r = strings.TrimSpace(r); r != "" {
				salesRecipients = append(salesRecipients, r)
			}
		}
	}
	if len(salesRecipients) == 0 {
		http.Error(w, "Sales email not configured", http.StatusInternalServerError)
		return
	}

	subject := "Enterprise Inquiry: " + req.Email
	html := `<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background-color:#f5f7fa;">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#f5f7fa;padding:40px 20px;">
<tr><td align="center">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="600" style="max-width:600px;background-color:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow:hidden;">
<tr><td style="padding:32px 40px;background:linear-gradient(135deg,#0052CC 0%,#003D99 100%);">
<h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:600;letter-spacing:-0.3px;">New Enterprise Inquiry</h1>
<p style="margin:4px 0 0 0;color:rgba(255,255,255,0.85);font-size:14px;">SchemaLabs Sales Dashboard</p>
</td></tr>
<tr><td style="padding:32px 40px;">
<p style="margin:0 0 24px 0;color:#172B4D;font-size:15px;line-height:1.5;">A potential enterprise customer has submitted a contact request. Details below:</p>
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:collapse;">
<tr><td style="padding:12px 0;border-bottom:1px solid #EBECF0;"><span style="color:#6B778C;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Work Email</span></td></tr>
<tr><td style="padding:4px 0 16px 0;color:#172B4D;font-size:15px;font-weight:500;">` + req.Email + `</td></tr>
<tr><td style="padding:12px 0;border-bottom:1px solid #EBECF0;"><span style="color:#6B778C;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Account Email</span></td></tr>
<tr><td style="padding:4px 0 16px 0;color:#172B4D;font-size:15px;">` + user.Email + `</td></tr>
<tr><td style="padding:12px 0;border-bottom:1px solid #EBECF0;"><span style="color:#6B778C;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Account Name</span></td></tr>
<tr><td style="padding:4px 0 16px 0;color:#172B4D;font-size:15px;">` + user.Name + `</td></tr>
<tr><td style="padding:12px 0;border-bottom:1px solid #EBECF0;"><span style="color:#6B778C;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Current Plan</span></td></tr>
<tr><td style="padding:4px 0 16px 0;"><span style="display:inline-block;padding:4px 12px;background-color:#DEEBFF;color:#0052CC;border-radius:4px;font-size:13px;font-weight:600;text-transform:uppercase;">` + currentPlan + `</span></td></tr>
<tr><td style="padding:12px 0;"><span style="color:#6B778C;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">User ID</span></td></tr>
<tr><td style="padding:4px 0 0 0;color:#6B778C;font-size:13px;font-family:'SF Mono',Monaco,Consolas,monospace;">` + userID + `</td></tr>
</table>
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-top:32px;">
<tr><td align="center">
<a href="mailto:` + req.Email + `" style="display:inline-block;padding:12px 32px;background-color:#0052CC;color:#ffffff;text-decoration:none;border-radius:6px;font-size:15px;font-weight:600;">Reply to Customer</a>
</td></tr>
</table>
</td></tr>
<tr><td style="padding:24px 40px;background-color:#FAFBFC;border-top:1px solid #EBECF0;">
<p style="margin:0;color:#6B778C;font-size:13px;text-align:center;">SchemaLabs, Inc. · Automated notification from sales system</p>
</td></tr>
</table>
</td></tr>
</table>
</body></html>`

	for _, rcpt := range salesRecipients {
		if err := svc.SendEmail(rcpt, subject, html); err != nil {
			log.Printf("[ENTERPRISE_INQUIRY] send failed to %s: %v", rcpt, err)
		}
	}

	confirmHTML := `<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background-color:#f5f7fa;">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#f5f7fa;padding:40px 20px;">
<tr><td align="center">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="600" style="max-width:600px;background-color:#ffffff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow:hidden;">
<tr><td style="padding:40px 40px 32px 40px;background:linear-gradient(135deg,#0052CC 0%,#003D99 100%);text-align:center;">
<h1 style="margin:0;color:#ffffff;font-size:26px;font-weight:600;letter-spacing:-0.5px;">Thanks for reaching out</h1>
<p style="margin:8px 0 0 0;color:rgba(255,255,255,0.9);font-size:15px;">Your Enterprise inquiry has been received.</p>
</td></tr>
<tr><td style="padding:40px;">
<p style="margin:0 0 20px 0;color:#172B4D;font-size:16px;line-height:1.6;">Hi there,</p>
<p style="margin:0 0 20px 0;color:#172B4D;font-size:16px;line-height:1.6;">Thank you for your interest in <strong>SchemaLabs Enterprise</strong>. Our sales team has received your request and will be in touch within <strong>1 business day</strong> to discuss your needs.</p>
<p style="margin:0 0 24px 0;color:#172B4D;font-size:16px;line-height:1.6;">In the meantime, here's what you can expect:</p>
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="margin:0 0 24px 0;">
<tr><td style="padding:16px;background-color:#F4F5F7;border-radius:8px;border-left:3px solid #0052CC;">
<p style="margin:0 0 4px 0;color:#172B4D;font-size:14px;font-weight:600;">Discovery Call</p>
<p style="margin:0;color:#6B778C;font-size:14px;line-height:1.5;">A 30-minute conversation to understand your use case and scale requirements.</p>
</td></tr>
<tr><td style="padding-top:12px;"></td></tr>
<tr><td style="padding:16px;background-color:#F4F5F7;border-radius:8px;border-left:3px solid #36B37E;">
<p style="margin:0 0 4px 0;color:#172B4D;font-size:14px;font-weight:600;">Custom Proposal</p>
<p style="margin:0;color:#6B778C;font-size:14px;line-height:1.5;">Tailored pricing, dedicated deployment, and SLA options for your organization.</p>
</td></tr>
<tr><td style="padding-top:12px;"></td></tr>
<tr><td style="padding:16px;background-color:#F4F5F7;border-radius:8px;border-left:3px solid #FFAB00;">
<p style="margin:0 0 4px 0;color:#172B4D;font-size:14px;font-weight:600;">Pilot & Onboarding</p>
<p style="margin:0;color:#6B778C;font-size:14px;line-height:1.5;">Hands-on setup with a dedicated Customer Success Manager.</p>
</td></tr>
</table>
<p style="margin:24px 0 0 0;color:#172B4D;font-size:16px;line-height:1.6;">Questions in the meantime? Reply to this email.</p>
<p style="margin:24px 0 0 0;color:#172B4D;font-size:16px;line-height:1.6;">— The SchemaLabs Team</p>
</td></tr>
<tr><td style="padding:24px 40px;background-color:#FAFBFC;border-top:1px solid #EBECF0;text-align:center;">
<p style="margin:0 0 4px 0;color:#6B778C;font-size:13px;">SchemaLabs, Inc.</p>
<p style="margin:0;color:#6B778C;font-size:12px;">Tabular foundation models for enterprise data.</p>
</td></tr>
</table>
</td></tr>
</table>
</body></html>`
	_ = svc.SendEmail(req.Email, "SchemaLabs Enterprise — request received", confirmHTML)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "sent"})
}
