package handlers

import (
"strings"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
)

type PredictRequest struct {
	Values [][]float64 `json:"values"`
	IncludeNarrative bool   `json:"include_narrative"`
}

type PredictResponse struct {
	Predictions   []int       `json:"predictions"`
	Confidences   []float64   `json:"confidences"`
	Probabilities [][]float64 `json:"probabilities"`
	Status        string      `json:"status"`
}

func PredictHandler(w http.ResponseWriter, r *http.Request) {
	predictStart := time.Now()
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

	// Check if narrative is requested
	if req.IncludeNarrative {
		var predResult map[string]interface{}
		if err := json.Unmarshal(body, &predResult); err == nil {
			narrative := generateNarrative(predResult, userID)
			if narrative != "" {
				predResult["language_output"] = map[string]interface{}{
					"narrative":     narrative,
					"model":         "mistral-medium-2505",
					"generation_ms": 0,
				}
				body, _ = json.Marshal(predResult)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
	InferenceRequestsTotal.WithLabelValues("success").Inc()
	InferenceDuration.Observe(time.Since(predictStart).Seconds())

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

func generateNarrative(predResult map[string]interface{}, userID string) string {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		return ""
	}

	start := time.Now()

	resultJSON, _ := json.Marshal(predResult)
	prompt := fmt.Sprintf("You are a data analyst. Given this prediction result, write a brief 2-3 sentence natural language explanation of what the model predicted, the confidence level, and key factors. Be concise and professional.\n\nPrediction Result:\n%s", string(resultJSON))

	reqBody := map[string]interface{}{
		"model": "mistral-medium-2505",
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 300,
	}

	jsonBody, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", "https://api.mistral.ai/v1/chat/completions", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(body, &result)

	choices, _ := result["choices"].([]interface{})
	if len(choices) == 0 {
		return ""
	}
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	narrative, _ := message["content"].(string)

	ms := time.Since(start).Milliseconds()
	fmt.Printf("[NARRATIVE] Generated in %dms for user %s\n", ms, userID)

	return narrative
}
