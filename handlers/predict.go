package handlers

import (
"strings"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type PredictRequest struct {
	Values [][]float64 `json:"values"`
}

type PredictResponse struct {
	Predictions   []int       `json:"predictions"`
	Confidences   []float64   `json:"confidences"`
	Probabilities [][]float64 `json:"probabilities"`
	Status        string      `json:"status"`
}

func PredictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

userID := r.Header.Get("X-User-ID")
if userID != "" {
var predErrors []string
if ok, reason := CheckQuota(userID, "query"); !ok {
predErrors = append(predErrors, reason)
}
if ok, reason := CheckCredits(userID, 0.05); !ok {
predErrors = append(predErrors, reason)
}
if len(predErrors) > 0 {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": strings.Join(predErrors, " | ")})
return
}
}

	jsonData, _ := json.Marshal(req)

	resp, err := http.Post(
		GetFlaskURL()+"/predict",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		http.Error(w, "Flask server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)

	// Deduct credits and log usage
	userID = r.Header.Get("X-User-ID")
	if userID != "" && DB != nil {
		var quota UserQuota
		if DB.Where("user_id = ?", userID).First(&quota).Error == nil {
			quota.CreditsUsed += 0.05
			DB.Save(&quota)
		}
		DB.Create(&UsageLog{
			ID: uuid.New().String(), UserID: userID, EventType: "predict", EventName: "Prediction",
			CreditsUsed: 0.05, ModelUsed: "schema-v0", CreatedAt: time.Now(),
		})
	}
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(GetFlaskURL()+"/health")
	if err != nil {
		http.Error(w, "Flask server down", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}

func ModelInfoHandler(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(GetFlaskURL()+"/model/info")
	if err != nil {
		http.Error(w, "Flask server down", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}
