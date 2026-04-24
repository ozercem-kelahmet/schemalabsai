package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"
)

type GiftCode struct {
	Code          string    `gorm:"primaryKey" json:"code"`
	Provider      string    `json:"provider"`
	TotalCredits  float64   `json:"total_credits"`
	UsedCredits   float64   `gorm:"default:0" json:"used_credits"`
	ValidUntil    time.Time `json:"valid_until"`
	RedeemedBy    string    `gorm:"index" json:"redeemed_by"`
	RedeemedAt    time.Time `json:"redeemed_at"`
	CreatedAt     time.Time `json:"created_at"`
}

type GiftCodeResponse struct {
	ID               string  `json:"id"`
	Code             string  `json:"code"`
	Provider         string  `json:"provider"`
	TotalCredits     float64 `json:"totalCredits"`
	UsedCredits      float64 `json:"usedCredits"`
	RemainingCredits float64 `json:"remainingCredits"`
	ValidUntil       string  `json:"validUntil"`
	RedeemedAt       string  `json:"redeemedAt"`
}

func ListGiftCodesHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var codes []GiftCode
	DB.Where("redeemed_by = ?", userID).Order("redeemed_at DESC").Find(&codes)

	out := make([]GiftCodeResponse, 0, len(codes))
	for _, c := range codes {
		out = append(out, GiftCodeResponse{
			ID:               "gc-" + c.Code,
			Code:             c.Code,
			Provider:         c.Provider,
			TotalCredits:     c.TotalCredits,
			UsedCredits:      c.UsedCredits,
			RemainingCredits: c.TotalCredits - c.UsedCredits,
			ValidUntil:       c.ValidUntil.Format("2006-01-02"),
			RedeemedAt:       c.RedeemedAt.Format("2006-01-02"),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func RedeemGiftCodeHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Code string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	code := strings.ToUpper(strings.TrimSpace(req.Code))
	if code == "" {
		http.Error(w, "Code required", http.StatusBadRequest)
		return
	}

	var gc GiftCode
	if err := DB.Where("code = ?", code).First(&gc).Error; err != nil {
		http.Error(w, "Invalid code", http.StatusNotFound)
		return
	}

	if gc.RedeemedBy != "" {
		http.Error(w, "Code already redeemed", http.StatusConflict)
		return
	}

	if !gc.ValidUntil.IsZero() && time.Now().After(gc.ValidUntil) {
		http.Error(w, "Code expired", http.StatusGone)
		return
	}

	if _, err := GetOrCreateQuota(userID); err != nil {
		http.Error(w, "Quota error", http.StatusInternalServerError)
		return
	}

	now := time.Now()
	r1 := DB.Exec("UPDATE gift_codes SET redeemed_by = ?, redeemed_at = ? WHERE code = ? AND (redeemed_by IS NULL OR redeemed_by = '')", userID, now, gc.Code)
	if r1.Error != nil || r1.RowsAffected == 0 {
		log.Printf("[GIFT] mark redeemed failed code=%s err=%v rows=%d", gc.Code, r1.Error, r1.RowsAffected)
		http.Error(w, "Code already redeemed", http.StatusConflict)
		return
	}

	r2 := DB.Exec("UPDATE user_quotas SET credits_balance_usd = credits_balance_usd + ?, credits_purchased_month = credits_purchased_month + ?, updated_at = NOW() WHERE user_id = ?", gc.TotalCredits, gc.TotalCredits, userID)
	if r2.Error != nil {
		log.Printf("[GIFT] credit update failed user=%s err=%v", userID, r2.Error)
		http.Error(w, "Credit update failed", http.StatusInternalServerError)
		return
	}
	log.Printf("[GIFT] redeemed code=%s user=%s amount=%.2f rows=%d", gc.Code, userID, gc.TotalCredits, r2.RowsAffected)

	var newBalance float64
	DB.Raw("SELECT credits_balance_usd FROM user_quotas WHERE user_id = ?", userID).Scan(&newBalance)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "success",
		"credits_added": gc.TotalCredits,
		"new_balance":   newBalance,
		"provider":      gc.Provider,
	})
}

func CreateGiftCodeHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		http.Error(w, "User not found", http.StatusUnauthorized)
		return
	}
	if user.Role != "admin" {
		http.Error(w, "Admin only", http.StatusForbidden)
		return
	}

	var req struct {
		Code          string  `json:"code"`
		Provider      string  `json:"provider"`
		TotalCredits  float64 `json:"total_credits"`
		ValidUntil    string  `json:"valid_until"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	code := strings.ToUpper(strings.TrimSpace(req.Code))
	if code == "" || req.Provider == "" || req.TotalCredits <= 0 {
		http.Error(w, "code, provider, total_credits required", http.StatusBadRequest)
		return
	}

	var existing GiftCode
	if err := DB.Where("code = ?", code).First(&existing).Error; err == nil {
		http.Error(w, "Code already exists", http.StatusConflict)
		return
	}

	validUntil := time.Now().AddDate(1, 0, 0)
	if req.ValidUntil != "" {
		if t, err := time.Parse("2006-01-02", req.ValidUntil); err == nil {
			validUntil = t
		}
	}

	gc := GiftCode{
		Code:         code,
		Provider:     req.Provider,
		TotalCredits: req.TotalCredits,
		ValidUntil:   validUntil,
		CreatedAt:    time.Now(),
	}
	if err := DB.Create(&gc).Error; err != nil {
		http.Error(w, "DB error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(gc)
}

func ListAllGiftCodesHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil || user.Role != "admin" {
		http.Error(w, "Admin only", http.StatusForbidden)
		return
	}

	var codes []GiftCode
	DB.Order("created_at DESC").Limit(500).Find(&codes)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(codes)
}
