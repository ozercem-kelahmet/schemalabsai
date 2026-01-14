package handlers

import (
	"bytes"
	"log"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

// sanitizeFileID - Path traversal önlemek için file ID sanitize et
func sanitizeFileID(id string) string {
	cleaned := ""
	for _, c := range id {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			cleaned += string(c)
		}
	}
	if cleaned == "" {
		return "invalid"
	}
	return cleaned
}

type TrainRequest struct {
	FileID       string `json:"file_id"`
	Filename     string `json:"filename"`
	Epochs       int    `json:"epochs"`
	BatchSize    int    `json:"batch_size"`
	TargetColumn string `json:"target_column,omitempty"`
}

type TrainResponse struct {
	JobID     string  `json:"job_id"`
	Status    string  `json:"status"`
	Message   string  `json:"message"`
	ModelName string  `json:"model_name"`
	ModelPath string  `json:"model_path"`
	ModelID   string  `json:"model_id"`
	Accuracy  float64 `json:"accuracy"`
	Rows      int     `json:"rows"`
	Epochs    int     `json:"epochs"`
	Loss      float64 `json:"loss"`
}

func TrainHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	var req TrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Epochs == 0 {
		req.Epochs = 5
	}
	if req.BatchSize == 0 {
		req.BatchSize = 64
	}

	pattern := "./uploads/" + sanitizeFileID(req.FileID) + "_*"
	matches, err := filepath.Glob(pattern)

	if err != nil || len(matches) == 0 {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}

	file, err := os.Open(matches[0])
	if err != nil {
		http.Error(w, "Failed to read file", http.StatusInternalServerError)
		return
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, _ := writer.CreateFormFile("file", filepath.Base(matches[0]))
	io.Copy(part, file)

	epochsField, _ := writer.CreateFormField("epochs")
	epochsField.Write([]byte(fmt.Sprintf("%d", req.Epochs)))

	batchField, _ := writer.CreateFormField("batch_size")
	batchField.Write([]byte(fmt.Sprintf("%d", req.BatchSize)))

	if req.TargetColumn != "" {
		targetField, _ := writer.CreateFormField("target_column")
		targetField.Write([]byte(req.TargetColumn))
	}

	writer.Close()

	resp, err := http.Post(
		GetFlaskURL()+"/finetune",
		writer.FormDataContentType(),
		body,
	)
	if err != nil {
		http.Error(w, "Flask server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(resp.Body)

	var flaskResp map[string]interface{}
	json.Unmarshal(responseBody, &flaskResp)

	now := time.Now()
	timestamp := now.Format("20060102_150405")
	
	baseName := req.Filename
	if baseName == "" {
		baseName = filepath.Base(matches[0])
		parts := strings.SplitN(baseName, "_", 2)
		if len(parts) > 1 {
			baseName = parts[1]
		}
	}
	baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))
	
	var versionCount int64
	if DB != nil && userID != "" {
		DB.Model(&FineTunedModel{}).Where("source_file_id = ? AND user_id = ?", req.FileID, userID).Count(&versionCount)
	}
	version := int(versionCount) + 1
	
	modelName := fmt.Sprintf("model_%s_%s_v%d", baseName, timestamp, version)
	
	modelPath := ""
	accuracy := 0.0
	if mp, ok := flaskResp["model_path"].(string); ok {
		modelPath = mp
	}
	if acc, ok := flaskResp["accuracy"].(float64); ok {
		accuracy = acc
	}
	loss := 0.0
	if l, ok := flaskResp["loss"].(float64); ok {
		loss = l
	}
modelID := ""
if mid, ok := flaskResp["model_id"].(string); ok {
modelID = mid
}

// Check for merged file ID from Flask
mergedFileID := ""
if mfid, ok := flaskResp["merged_file_id"].(string); ok && mfid != "" {
mergedFileID = mfid
}

	if DB != nil && userID != "" {
		ftModel := FineTunedModel{
			ID:           uuid.New().String(),
			Name:         modelName,
			Version:      version,
			SourceFileID: func() string { if mergedFileID != "" { return mergedFileID }; return req.FileID }(),
			SourceName:   baseName,
			ModelPath:    modelPath,
			Accuracy:     accuracy,
			Epochs:       req.Epochs,
			BatchSize:    req.BatchSize,
		Loss:         loss,
			UserID:       userID,
			CreatedAt:    now,
		}
		DB.Create(&ftModel)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(TrainResponse{
		JobID:     uuid.New().String(),
		Status:    "success",
		Message:   "Model trained successfully",
		ModelName: modelName,
		ModelPath: modelPath,
ModelID:   modelID,
		Accuracy:  accuracy,
	})
}

func ListFineTunedModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var models []FineTunedModel
	DB.Where("user_id = ?", userID).Order("created_at desc").Find(&models)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"models": models})
}

type MultiTrainRequest struct {
	FileIDs      []string `json:"file_ids"`
	ModelName    string   `json:"model_name"`
	Epochs       int      `json:"epochs"`
	BatchSize    int      `json:"batch_size"`
	LearningRate float64  `json:"learning_rate"`
	WarmupSteps  int      `json:"warmup_steps"`
QueryID      string   `json:"query_id"`
}

func MultiTrainHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	var req MultiTrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Epochs == 0 {
		req.Epochs = 5
	}
	if req.BatchSize == 0 {
		req.BatchSize = 64
	}
	if req.LearningRate == 0 {
		req.LearningRate = 0.001
	}

	// Collect all file paths
	var filePaths []string
	for _, fileID := range req.FileIDs {
		pattern := "./uploads/" + fileID + "_*"
		matches, err := filepath.Glob(pattern)
		if err != nil || len(matches) == 0 {
			continue
		}
		filePaths = append(filePaths, matches[0])
	}

	if len(filePaths) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "No files found. Files may not exist locally or need to be re-uploaded.",
		})
		return
	}

	// Create multipart form with multiple files
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	for _, filePath := range filePaths {
		file, err := os.Open(filePath)
		if err != nil {
			continue
		}
		fieldName := "file"
		part, _ := writer.CreateFormFile(fieldName, filepath.Base(filePath))
		io.Copy(part, file)
		file.Close()
	}

	// Add training parameters
	epochsField, _ := writer.CreateFormField("epochs")
	epochsField.Write([]byte(fmt.Sprintf("%d", req.Epochs)))

	batchField, _ := writer.CreateFormField("batch_size")
	batchField.Write([]byte(fmt.Sprintf("%d", req.BatchSize)))

	lrField, _ := writer.CreateFormField("learning_rate")
	lrField.Write([]byte(fmt.Sprintf("%f", req.LearningRate)))

	warmupField, _ := writer.CreateFormField("warmup_steps")
	warmupField.Write([]byte(fmt.Sprintf("%d", req.WarmupSteps)))

queryIDField, _ := writer.CreateFormField("query_id")
queryIDField.Write([]byte(req.QueryID))

	mergeField, _ := writer.CreateFormField("merge_files")
	mergeField.Write([]byte("true"))

	writer.Close()

	// Call Flask server with timeout
	httpClient := &http.Client{Timeout: 18000 * time.Second}
	httpReq, _ := http.NewRequest("POST", GetFlaskURL()+"/finetune", body)
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		http.Error(w, "Flask server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(resp.Body)

	var flaskResp map[string]interface{}
	json.Unmarshal(responseBody, &flaskResp)

	now := time.Now()
	timestamp := now.Format("20060102_150405")

	modelName := req.ModelName
	if modelName == "" {
		modelName = fmt.Sprintf("model_merged_%s", timestamp)
	}

	modelPath := ""
	accuracy := 0.0
	if mp, ok := flaskResp["model_path"].(string); ok {
		modelPath = mp
	}
	if acc, ok := flaskResp["accuracy"].(float64); ok {
		accuracy = acc
	}
	loss := 0.0
	if l, ok := flaskResp["loss"].(float64); ok {
		loss = l
	}

	// Save to database
	if DB != nil && userID != "" {
		ftModel := FineTunedModel{
			ID:           uuid.New().String(),
			Name:         modelName,
			Version:      1,
			SourceFileID: strings.Join(req.FileIDs, ","),
			SourceName:   fmt.Sprintf("%d files merged", len(req.FileIDs)),
			ModelPath:    modelPath,
			Accuracy:     accuracy,
			Epochs:       req.Epochs,
			BatchSize:    req.BatchSize,
		Loss:         loss,
			UserID:       userID,
			CreatedAt:    now,
		}
		DB.Create(&ftModel)
	}

	w.Header().Set("Content-Type", "application/json")
	rows := 0
	if r, ok := flaskResp["rows"].(float64); ok {
		rows = int(r)
	}
	epochs := 0
	if e, ok := flaskResp["epochs"].(float64); ok {
		epochs = int(e)
	}
	
modelID := ""
if mid, ok := flaskResp["model_id"].(string); ok {
modelID = mid
}

	json.NewEncoder(w).Encode(TrainResponse{
		JobID:     uuid.New().String(),
		Status:    "success",
		Message:   fmt.Sprintf("Model trained with %d merged files", len(filePaths)),
		ModelName: modelName,
		ModelPath: modelPath,
ModelID:   modelID,
		Accuracy:  accuracy,
		Rows:      rows,
		Epochs:    epochs,
		Loss:      loss,
	})
}

func DeleteFineTunedModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	modelID := r.URL.Query().Get("id")
	
	if modelID == "" {
		// Extract from path: /api/models/finetuned/{id}
		path := r.URL.Path
		parts := strings.Split(path, "/")
		if len(parts) > 0 {
			modelID = parts[len(parts)-1]
		}
	}

	if userID == "" || modelID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if DB != nil {
		DB.Where("id = ? AND user_id = ?", modelID, userID).Delete(&FineTunedModel{})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

type AnalyzeRequest struct {
	FileIDs []string `json:"file_ids"`
}

func AnalyzeFilesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var filePaths []string
	for _, fileID := range req.FileIDs {
		pattern := "./uploads/" + fileID + "_*"
		matches, err := filepath.Glob(pattern)
		if err != nil || len(matches) == 0 {
			continue
		}
		filePaths = append(filePaths, matches[0])
	}

	if len(filePaths) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "No files found. Files may not exist locally or need to be re-uploaded.",
		})
		return
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	for _, filePath := range filePaths {
		file, err := os.Open(filePath)
		if err != nil {
			continue
		}
		fieldName := "file"
		part, _ := writer.CreateFormFile(fieldName, filepath.Base(filePath))
		io.Copy(part, file)
		file.Close()
	}

	analyzeField, _ := writer.CreateFormField("analyze_only")
	analyzeField.Write([]byte("true"))
	writer.Close()

	resp, err := http.Post(
		GetFlaskURL()+"/finetune",
		writer.FormDataContentType(),
		body,
	)
	if err != nil {
		http.Error(w, "Flask server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(resp.Body)
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseBody)
}

var trainingProgress = struct {
	Epoch    int     `json:"epoch"`
	Epochs   int     `json:"epochs"`
	Accuracy float64 `json:"accuracy"`
	Loss     float64 `json:"loss"`
	Status   string  `json:"status"`
}{}

func TrainingProgressHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(trainingProgress)
}

func UpdateFineTunedModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var model FineTunedModel
	if err := DB.Where("id = ? AND user_id = ?", req.ID, userID).First(&model).Error; err != nil {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	model.Name = req.Name
	DB.Save(&model)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Model updated"})
}

// AsyncTrainHandler - Async training başlatır
func AsyncTrainHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		FileIDs       []string `json:"file_ids"`
		ModelName     string   `json:"model_name"`
		Epochs        int      `json:"epochs"`
		BatchSize     int      `json:"batch_size"`
		LearningRate  float64  `json:"learning_rate"`
		WarmupSteps   int      `json:"warmup_steps"`
		QueryID       string   `json:"query_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	flaskURL := GetFlaskURL() + "/finetune/async"
	
	// Flask'a yönlendir
	resp, err := http.Post(flaskURL, "application/json", bytes.NewBuffer(mustMarshal(req)))
	if err != nil {
		log.Printf("Flask error: %v", err)
	http.Error(w, "Flask error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

// TrainingStatusHandler - Training status döner
func TrainingStatusHandler(w http.ResponseWriter, r *http.Request) {
	taskID := r.URL.Query().Get("task_id")
	if taskID == "" {
		http.Error(w, "task_id required", http.StatusBadRequest)
		return
	}

	flaskURL := GetFlaskURL() + "/training/status/" + taskID
	
	resp, err := http.Get(flaskURL)
	if err != nil {
		log.Printf("Flask error: %v", err)
	http.Error(w, "Flask error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

func mustMarshal(v interface{}) []byte {
	b, _ := json.Marshal(v)
	return b
}
