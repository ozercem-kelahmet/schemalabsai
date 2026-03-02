package handlers

import (
"strings"
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/google/uuid"
)

// AnalyzeHandler - API Key ile dosya analizi (CSV, Excel, JSON)
// LLM yok, sadece base model analizi döner
func AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// API Key middleware'den user gelir
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, `{"error": "Unauthorized"}`, http.StatusUnauthorized)
		return
	}

	// Multipart form parse (32MB max)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, `{"error": "Invalid form data or file too large"}`, http.StatusBadRequest)
		return
	}


// Check quota
var analyzeErrors []string
if ok, reason := CheckQuota(userID, "query"); !ok {
analyzeErrors = append(analyzeErrors, reason)
}
if ok, reason := CheckCredits(userID, 0.10); !ok {
analyzeErrors = append(analyzeErrors, reason)
}
if len(analyzeErrors) > 0 {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": strings.Join(analyzeErrors, " | ")})
return
}

	query := r.FormValue("query")
	if query == "" {
		query = "Analyze this data"
	}

	// Dosyayı al
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, `{"error": "No file provided. Send file as multipart form-data"}`, http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Flask'a multipart olarak gönder
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// File ekle
	part, _ := writer.CreateFormFile("file", header.Filename)
	io.Copy(part, file)

	// Query ve model_id ekle
	writer.WriteField("query", query)
	writer.WriteField("model_id", "schema-v0")
	writer.WriteField("user_id", userID)

	writer.Close()

	// Flask /analyze_file endpoint'ine gönder
	flaskURL := GetFlaskURL() + "/analyze_file"
	req, _ := http.NewRequest("POST", flaskURL, body)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, `{"error": "Analysis service unavailable"}`, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	// Flask response'u döndür
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)

	// Deduct credits and log usage
	if DB != nil {
		var quota UserQuota
		if DB.Where("user_id = ?", userID).First(&quota).Error == nil {
			quota.CreditsUsed += 0.10
			DB.Save(&quota)
		}
		DB.Create(&UsageLog{
			ID: uuid.New().String(), UserID: userID, EventType: "analyze", EventName: "Data Analysis",
			CreditsUsed: 0.10, ModelUsed: "schema-v0", CreatedAt: time.Now(),
		})
	}
}

// AnalyzeEndpointHandler - /v1/analyze/{path} - Analyze endpoint with file upload
func AnalyzeEndpointHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/v1/analyze/")
	if path == "" {
		http.Error(w, `{"error": "Missing endpoint path"}`, http.StatusBadRequest)
		return
	}

	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		http.Error(w, `{"error": "Missing Authorization header"}`, http.StatusUnauthorized)
		return
	}
	apiKey := strings.TrimPrefix(authHeader, "Bearer ")

	var key APIKey
	if DB.Where("key = ?", apiKey).First(&key).Error != nil {
		http.Error(w, `{"error": "Invalid API key"}`, http.StatusUnauthorized)
		return
	}

	var endpoint Endpoint
	if DB.Where("path = ? AND user_id = ?", path, key.UserID).First(&endpoint).Error != nil {
		http.Error(w, `{"error": "Endpoint not found"}`, http.StatusNotFound)
		return
	}

	if ok, reason := CheckCredits(key.UserID, 0.10); !ok {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]string{"error": reason})
		return
	}

	// Parse multipart
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, `{"error": "Invalid form data"}`, http.StatusBadRequest)
		return
	}

	query := r.FormValue("query")
	if query == "" {
		query = "Analyze this data"
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, `{"error": "No file provided"}`, http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Build multipart for Flask
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	part, _ := mw.CreateFormFile("file", header.Filename)
	io.Copy(part, file)
	mw.WriteField("query", query)
	mw.WriteField("model_id", "schema-v0")
	mw.WriteField("user_id", endpoint.UserID)
	if endpoint.VerticalConfigID != "" {
		mw.WriteField("vertical_config_id", endpoint.VerticalConfigID)
	}
	mw.Close()

	flaskURL := GetFlaskURL() + "/analyze_file"
	req2, _ := http.NewRequest("POST", flaskURL, body)
	req2.Header.Set("Content-Type", mw.FormDataContentType())

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req2)
	if err != nil {
		http.Error(w, `{"error": "Analysis service unavailable"}`, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	// Pass through Flask response + add endpoint info
	var flaskResponse map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&flaskResponse)

	result := map[string]interface{}{
		"query":       query,
		"endpoint_id": endpoint.ID,
		"endpoint_type": "analyze",
	}
	for k, v := range flaskResponse {
		result[k] = v
	}

	// Add vertical info
	if endpoint.VerticalConfigID != "" {
		var vc VerticalConfig
		if DB.Where("id = ?", endpoint.VerticalConfigID).First(&vc).Error == nil {
			vInfo := map[string]interface{}{
				"id": vc.ID, "name": vc.Name, "enabled": vc.Enabled, "config": vc.ConfigYAML,
			}
			var tools []VerticalTool
			DB.Where("vertical_id = ? AND user_id = ?", vc.ID, endpoint.UserID).Find(&tools)
			tList := []map[string]string{}
			for _, t := range tools {
				tList = append(tList, map[string]string{"name": t.Name, "hook": t.Hook, "status": t.ValidationStatus})
			}
			vInfo["tools"] = tList
			var agents []VerticalAgent
			DB.Where("vertical_id = ? AND user_id = ?", vc.ID, endpoint.UserID).Find(&agents)
			aList := []map[string]string{}
			for _, a := range agents {
				aList = append(aList, map[string]string{"name": a.Name, "role": a.Role, "status": a.ValidationStatus})
			}
			vInfo["agents"] = aList
			result["vertical"] = vInfo
		}
	}

	DB.Model(&endpoint).Update("calls", endpoint.Calls+1)
	DB.Model(&key).Updates(map[string]interface{}{"requests": key.Requests + 1, "last_used": time.Now()})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)

	// Credits
	if DB != nil {
		var quota UserQuota
		if DB.Where("user_id = ?", key.UserID).First(&quota).Error == nil {
			quota.CreditsUsed += 0.10
			DB.Save(&quota)
		}
		DB.Create(&UsageLog{
			ID: uuid.New().String(), UserID: key.UserID, EventType: "analyze_endpoint",
			EventName: "Analyze Endpoint: " + endpoint.Name, CreditsUsed: 0.10,
			ModelUsed: "schema-v0", CreatedAt: time.Now(),
		})
	}
}
