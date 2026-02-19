package handlers

import (
	"bytes"
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
