package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"schemalabsai/services"
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
estInRows := len(req.Values)
estInCols := 0
if estInRows > 0 {
estInCols = len(req.Values[0])
}
if ok, reason := CheckRateLimit(userID, RateLimitSchema, int64(estInRows*estInCols), int64(estInRows)); !ok {
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

	userID = r.Header.Get("X-User-ID")
	if userID != "" && DB != nil {
		rows := len(req.Values)
		cols := 0
		if rows > 0 {
			cols = len(req.Values[0])
		}
		outputRows := rows
		if err := TrackSchemaCall(userID, rows, cols, outputRows, getBaseModelVersion(), false); err != nil {
			log.Printf("[PREDICT] TrackSchemaCall failed for user %s: %v", userID, err)
		}
		DB.Create(&UsageLog{
			ID: uuid.New().String(), UserID: userID, EventType: "predict", EventName: "Prediction",
			CreditsUsed: 0, ModelUsed: getBaseModelID(), CreatedAt: time.Now(),
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

// BatchPredictHandler - Spark ile 1M+ satır toplu tahmin
func BatchPredictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	// Multipart file upload
	if err := r.ParseMultipartForm(500 << 20); err != nil {
		http.Error(w, "File too large", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "No file provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	modelID := r.FormValue("model_id")
	if modelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}

	// Quota check
	if userID != "" {
		if ok, reason := CheckQuota(userID, "query"); !ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			json.NewEncoder(w).Encode(map[string]string{"error": reason})
			return
		}
	}

	// Temp dosyaya kaydet
	tmpPath := fmt.Sprintf("./uploads/batch_%s_%s.csv", userID, uuid.New().String()[:8])
	tmpFile, err := os.Create(tmpPath)
	if err != nil {
		http.Error(w, "Failed to save file", http.StatusInternalServerError)
		return
	}
	io.Copy(tmpFile, file)
	tmpFile.Close()
	defer os.Remove(tmpPath)

	log.Printf("[BATCH PREDICT] File: %s size=%.1fMB model=%s", header.Filename, float64(fileSize(tmpPath))/(1024*1024), modelID)

	// Spark ile büyük dosyaysa preprocess
	outputPath := strings.TrimSuffix(tmpPath, ".csv") + "_result.csv"
	if services.DefaultSpark != nil && services.DefaultSpark.IsAvailable() && services.DefaultSpark.ShouldUseSparkBySize(fileSize(tmpPath)) {
		job := services.SparkJobRequest{
			JobType:    "preprocess",
			OutputPath: outputPath,
			Config:     map[string]string{"input_paths": tmpPath},
		}
		resp, err := services.DefaultSpark.SubmitJob(job)
		if err == nil {
			result, werr := services.DefaultSpark.WaitForJob(resp.JobID, 10*60*1000000000)
			if werr == nil && result.Status == "completed" {
				tmpPath = outputPath
				log.Printf("[BATCH PREDICT] Spark preprocessed: %d rows", result.RowCount)
			}
		}
	}

	// Flask'a batch predict gönder
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("file", filepath.Base(tmpPath))
	f, _ := os.Open(tmpPath)
	io.Copy(part, f)
	f.Close()
	writer.WriteField("model_id", modelID)
	writer.WriteField("batch_mode", "true")
	writer.Close()

	flaskResp, err := http.Post(GetFlaskURL()+"/batch_predict", writer.FormDataContentType(), body)
	if err != nil {
		http.Error(w, "Flask error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer flaskResp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, flaskResp.Body)
}

func fileSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
