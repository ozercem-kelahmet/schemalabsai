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
	"context"
	"database/sql"
	"encoding/csv"
	"gorm.io/gorm"
	"gorm.io/driver/postgres"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"net"
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
	log.Printf("=== TRAIN HANDLER START: user=%s ===", userID)

	var req TrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}


if req.Epochs == 0 {
		req.Epochs = 5
	}

// Check quota
allowed, reason := CheckQuota(userID, "train")
if !allowed {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": reason})
return
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
	log.Printf("Flask response body: %s", string(responseBody))

// Fix NaN/Infinity in JSON (invalid JSON values from Python)
cleanBody := strings.ReplaceAll(string(responseBody), ": NaN", ": 0")
cleanBody = strings.ReplaceAll(cleanBody, ":NaN", ":0")
cleanBody = strings.ReplaceAll(cleanBody, "NaN", "0")
cleanBody = strings.ReplaceAll(cleanBody, "Infinity", "0")
cleanBody = strings.ReplaceAll(cleanBody, "-Infinity", "0")

	var flaskResp map[string]interface{}
json.Unmarshal([]byte(cleanBody), &flaskResp)
	log.Printf("Flask parsed response: %+v", flaskResp)

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

// Get actual epochs from Flask
if actualEpochs, ok := flaskResp["epochs"]; ok {
if ep, ok := actualEpochs.(float64); ok && ep > 0 {
req.Epochs = int(ep)
}
}
		accuracy = acc
		log.Printf("✅ Accuracy parsed from Flask: %.2f", accuracy)
	} else {
		log.Printf("⚠️  No accuracy in Flask response or type mismatch")
	}
	loss := 0.0
	if l, ok := flaskResp["loss"].(float64); ok {
		loss = l
	}

// Check for merged file ID from Flask
mergedFileID := ""
if mfid, ok := flaskResp["merged_file_id"].(string); ok && mfid != "" {
mergedFileID = mfid
}


	// Save merged file to uploaded_files table
	if mergedFileID != "" && userID != "" {
		mergedFilePath := "uploads/" + mergedFileID + ".csv"
		fileInfo, _ := os.Stat(mergedFilePath)
		fileSize := int64(0)
		if fileInfo != nil { fileSize = fileInfo.Size() }
		uploadedFile := UploadedFile{
			ID:        mergedFileID + ".csv",
			UserID:    userID,
			Filename:  mergedFileID + "_merged_all.csv",
			Path:      mergedFilePath,
			Size:      fileSize,
			CreatedAt: time.Now(),
IsMerged:  true,
		}
		DB.Create(&uploadedFile)
	}
	var dbModelID string
if DB != nil && userID != "" {
		dbModelID = uuid.New().String()
ftModel := FineTunedModel{
			ID:           dbModelID,
			Name:         modelName,
			Version:      version,
			SourceFileID: func() string { if mergedFileID != "" { return mergedFileID }; return req.FileID }(),
			SourceName:   baseName,
			SourceFiles:  req.FileID,
			ModelPath:    modelPath,
			Accuracy:     accuracy,
			Epochs:       func() int { if e, ok := flaskResp["epochs"].(float64); ok && e > 0 { return int(e) }; return req.Epochs }(),
			BatchSize:    req.BatchSize,
		Loss:         loss,
			UserID:       userID,
			CreatedAt:    now,
		}
		DB.Create(&ftModel)

// Deduct credits and log usage
UseCredit(userID, "train")
DB.Create(&UsageLog{
ID:           generateSessionID()[:16],
UserID:       userID,
EventType:    "train",
EventName:    "Model Training",
ResourceID:   ftModel.ID,
ResourceName: ftModel.Name,
CreditsUsed:  CreditPerTrain,
ModelUsed:    "schema-v0",
CreatedAt:    time.Now(),
})

	}

	// Send training complete email (only if training succeeded)
	if accuracy > 0 {
	var user User
	if DB.Where("id = ?", userID).First(&user).Error == nil {
		emailService := NewEmailService()
		emailService.SendTrainingComplete(user.Email, modelName, accuracy)
	}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(TrainResponse{
		JobID:     uuid.New().String(),
		Status:    "success",
		Message:   "Model trained successfully",
		ModelName: modelName,
		ModelPath: modelPath,
ModelID:   dbModelID,
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

	// Collect all unique file IDs
	fileIDSet := make(map[string]bool)
	for _, m := range models {
		if m.SourceFiles != "" {
			for _, fid := range strings.Split(m.SourceFiles, ",") {
				fid = strings.TrimSpace(fid)
				if fid != "" {
					fileIDSet[fid] = true
				}
			}
		}
	}

	// Fetch all files in one query
	var fileIDs []string
	for fid := range fileIDSet {
		fileIDs = append(fileIDs, fid)
	}
	
	fileNameMap := make(map[string]string)
	if len(fileIDs) > 0 {
		var files []UploadedFile
		DB.Where("id IN ?", fileIDs).Find(&files)
		for _, f := range files {
			fileNameMap[f.ID] = f.Filename
		}
	}

	// Build response
	var response []map[string]interface{}
	for _, m := range models {
		mr := map[string]interface{}{
			"id":             m.ID,
			"name":           m.Name,
			"version":        m.Version,
			"source_file_id": m.SourceFileID,
			"source_name":    m.SourceName,
			"source_files":   m.SourceFiles,
			"model_path":     m.ModelPath,
			"accuracy":       m.Accuracy,
			"epochs":         m.Epochs,
			"batch_size":     m.BatchSize,
			"loss":           m.Loss,
			"user_id":        m.UserID,
			"created_at":     m.CreatedAt,
"sync_mode":      m.SyncMode,
"sync_status":    m.SyncStatus,
"schedule_cron":  m.ScheduleCron,
"schedule_desc":  m.ScheduleDesc,
"next_sync_at":   m.NextSyncAt,
"last_sync_at":   m.LastSyncAt,
"connection_ids": m.ConnectionIDs,
		}

		if m.SourceFiles != "" {
			var fileNames []string
			for _, fid := range strings.Split(m.SourceFiles, ",") {
				fid = strings.TrimSpace(fid)
				if name, ok := fileNameMap[fid]; ok {
					fileNames = append(fileNames, name)
				}
			}
			mr["source_file_names"] = strings.Join(fileNames, ",")
		}
		response = append(response, mr)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"models": response})
}
type MultiTrainRequest struct {
	FileIDs      []string `json:"file_ids"`
	ModelName    string   `json:"model_name"`
	Epochs       int      `json:"epochs"`
	BatchSize    int      `json:"batch_size"`
	LearningRate float64  `json:"learning_rate"`
	WarmupSteps  int      `json:"warmup_steps"`
QueryID       string   `json:"query_id"`
SyncMode      string   `json:"sync_mode"`
ScheduleCron  string   `json:"schedule_cron"`
ScheduleDesc  string   `json:"schedule_desc"`
ConnectionIDs string   `json:"connection_ids"`
}



// convertJSONToCSV converts a JSON or JSONL file to CSV format
func convertJSONToCSV(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil { return "", err }

	var records []map[string]interface{}

	if strings.HasSuffix(filePath, ".jsonl") {
		// JSONL: one JSON object per line
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" { continue }
			var obj map[string]interface{}
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				// Try as array element
				continue
			}
			records = append(records, obj)
		}
	} else {
		// JSON: try array first, then single object, then nested
		if err := json.Unmarshal(data, &records); err != nil {
			// Try as {"data": [...]} or {"records": [...]}
			var wrapper map[string]interface{}
			if err2 := json.Unmarshal(data, &wrapper); err2 != nil {
				return "", fmt.Errorf("cannot parse JSON: %v", err)
			}
			// Find first array in wrapper
			for _, v := range wrapper {
				if arr, ok := v.([]interface{}); ok {
					for _, item := range arr {
						if m, ok := item.(map[string]interface{}); ok {
							records = append(records, m)
						}
					}
					break
				}
			}
			if len(records) == 0 {
				// Single object - flatten to one row
				records = append(records, wrapper)
			}
		}
	}

	if len(records) == 0 { return "", fmt.Errorf("no records found in %s", filePath) }

	// Collect all unique keys maintaining order
	keyMap := make(map[string]bool)
	var keys []string
	for _, rec := range records {
		for k := range rec {
			if !keyMap[k] {
				keyMap[k] = true
				keys = append(keys, k)
			}
		}
	}

	// Write CSV
	csvPath := strings.TrimSuffix(filePath, filepath.Ext(filePath)) + "_converted.csv"
	csvFile, err := os.Create(csvPath)
	if err != nil { return "", err }
	csvWriter := csv.NewWriter(csvFile)
	csvWriter.Write(keys)
	for _, rec := range records {
		row := make([]string, len(keys))
		for i, k := range keys {
			if v, ok := rec[k]; ok && v != nil {
				switch val := v.(type) {
				case string:
					row[i] = val
				case float64:
					if val == float64(int(val)) {
						row[i] = fmt.Sprintf("%d", int(val))
					} else {
						row[i] = fmt.Sprintf("%v", val)
					}
				default:
					b, _ := json.Marshal(val)
					row[i] = string(b)
				}
			}
		}
		csvWriter.Write(row)
	}
	csvWriter.Flush()
	csvFile.Close()
	log.Printf("Converted %s to CSV: %d records, %d columns", filePath, len(records), len(keys))
	return csvPath, nil
}

// exportConnectionToCSV connects to any supported database and exports tables as CSV files
func exportConnectionToCSV(conn Connection, connID string) ([]string, error) {
	var filePaths []string

	switch conn.SubType {
	case "postgresql", "supabase":
		connHost := conn.Host
		sslmode := "disable"
		if conn.SubType == "supabase" {
			addrs, _ := net.LookupIP(conn.Host)
			for _, ip := range addrs {
				if ip.To4() != nil { connHost = ip.String(); break }
			}
			sslmode = "require"
		}
		if conn.SSL { sslmode = "require" }
		dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s",
			conn.Username, conn.Password, connHost, conn.Port, conn.Database, sslmode)
		paths, err := exportSQLToCSV(dsn, "postgres", connID, "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'", true)
		if err != nil { return nil, err }
		filePaths = append(filePaths, paths...)

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		paths, err := exportSQLToCSV(dsn, "mysql", connID, fmt.Sprintf("SELECT table_name FROM information_schema.tables WHERE table_schema = '%s'", conn.Database), false)
		if err != nil { return nil, err }
		filePaths = append(filePaths, paths...)

	case "snowflake":
		dsn := fmt.Sprintf("postgresql://%s:%s@%s/%s",
			conn.Username, conn.Password, conn.Host, conn.Database)
		paths, err := exportSQLToCSV(dsn, "snowflake", connID, "SELECT table_name FROM information_schema.tables WHERE table_schema = 'PUBLIC'", true)
		if err != nil { return nil, err }
		filePaths = append(filePaths, paths...)

	case "mongodb":
		uri := fmt.Sprintf("mongodb://%s:%s@%s:%d", conn.Username, conn.Password, conn.Host, conn.Port)
		if conn.Username == "" { uri = fmt.Sprintf("mongodb://%s:%d", conn.Host, conn.Port) }
		if conn.SSL { uri += "/?tls=true" }
		ctx := context.Background()
		clientOpts := options.Client().ApplyURI(uri).SetConnectTimeout(10 * time.Second)
		client, err := mongo.Connect(ctx, clientOpts)
		if err != nil { return nil, fmt.Errorf("mongo connect failed: %v", err) }
		defer client.Disconnect(ctx)
		db := client.Database(conn.Database)
		collections, _ := db.ListCollectionNames(ctx, map[string]interface{}{})
		for _, collName := range collections {
			csvPath := exportMongoToCSV(ctx, db, collName, conn, connID)
			if csvPath != "" { filePaths = append(filePaths, csvPath) }
		}

	case "databricks":
		dsn := fmt.Sprintf("postgresql://%s:%s@%s/%s",
			conn.Username, conn.Password, conn.Host, conn.Database)
		paths, err := exportSQLToCSV(dsn, "databricks", connID, "SHOW TABLES", false)
		if err != nil { log.Printf("Databricks export failed: %v", err) }
		if len(paths) > 0 { filePaths = append(filePaths, paths...) }

	case "rest_api":
		paths, err := exportAPIToCSV(conn, connID)
		if err != nil { log.Printf("REST API export failed: %v", err) }
		if len(paths) > 0 { filePaths = append(filePaths, paths...) }

	case "graphql":
		paths, err := exportGraphQLToCSV(conn, connID)
		if err != nil { log.Printf("GraphQL export failed: %v", err) }
		if len(paths) > 0 { filePaths = append(filePaths, paths...) }

	case "pinecone":
		if conn.Endpoint != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			listURL := conn.Endpoint + "/vectors/list?limit=10000"
			req, _ := http.NewRequest("GET", listURL, nil)
			req.Header.Set("Api-Key", conn.APIKey)
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("pinecone failed: %v", err) }
			defer resp.Body.Close()
			var result struct {
				Vectors []struct {
					ID       string                 `json:"id"`
					Metadata map[string]interface{} `json:"metadata"`
				} `json:"vectors"`
			}
			json.NewDecoder(resp.Body).Decode(&result)
			if len(result.Vectors) > 0 {
				csvPath := fmt.Sprintf("./uploads/conn_%s_pinecone.csv", connID)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				headers := []string{"id"}
				if result.Vectors[0].Metadata != nil {
					for k := range result.Vectors[0].Metadata { headers = append(headers, k) }
				}
				csvWriter.Write(headers)
				for _, v := range result.Vectors {
					row := []string{v.ID}
					for _, h := range headers[1:] { row = append(row, fmt.Sprintf("%v", v.Metadata[h])) }
					csvWriter.Write(row)
				}
				csvWriter.Flush()
				csvFile.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported Pinecone %d vectors to %s", len(result.Vectors), csvPath)
			}
		}

	case "weaviate":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// Get all classes first
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/v1/schema", nil)
			if conn.APIKey != "" { req.Header.Set("Authorization", "Bearer "+conn.APIKey) }
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("weaviate failed: %v", err) }
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			csvPath := fmt.Sprintf("./uploads/conn_%s_weaviate.csv", connID)
			csvFile, _ := os.Create(csvPath)
			csvWriter := csv.NewWriter(csvFile)
			csvWriter.Write([]string{"schema_data"})
			csvWriter.Write([]string{string(bodyBytes)})
			csvWriter.Flush()
			csvFile.Close()
			filePaths = append(filePaths, csvPath)
			log.Printf("Exported Weaviate schema to %s", csvPath)
		}

	case "chroma":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/api/v1/collections", nil)
			if conn.APIKey != "" { req.Header.Set("Authorization", "Bearer "+conn.APIKey) }
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("chroma failed: %v", err) }
			defer resp.Body.Close()
			var collections []struct { ID string `json:"id"`; Name string `json:"name"` }
			json.NewDecoder(resp.Body).Decode(&collections)
			for _, coll := range collections {
				getBody, _ := json.Marshal(map[string]interface{}{"limit": 10000, "include": []string{"documents", "metadatas"}})
				getReq, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/api/v1/collections/"+coll.ID+"/get", bytes.NewReader(getBody))
				getReq.Header.Set("Content-Type", "application/json")
				if conn.APIKey != "" { getReq.Header.Set("Authorization", "Bearer "+conn.APIKey) }
				getResp, gerr := httpClient.Do(getReq)
				if gerr != nil { continue }
				var getResult struct {
					IDs       []string                 `json:"ids"`
					Documents []string                 `json:"documents"`
					Metadatas []map[string]interface{} `json:"metadatas"`
				}
				json.NewDecoder(getResp.Body).Decode(&getResult)
				getResp.Body.Close()
				if len(getResult.IDs) == 0 { continue }
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, coll.Name)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				headers := []string{"id", "document"}
				if len(getResult.Metadatas) > 0 && getResult.Metadatas[0] != nil {
					for k := range getResult.Metadatas[0] { headers = append(headers, k) }
				}
				csvWriter.Write(headers)
				for i, id := range getResult.IDs {
					row := []string{id}
					if i < len(getResult.Documents) { row = append(row, getResult.Documents[i]) } else { row = append(row, "") }
					if i < len(getResult.Metadatas) && getResult.Metadatas[i] != nil {
						for _, h := range headers[2:] { row = append(row, fmt.Sprintf("%v", getResult.Metadatas[i][h])) }
					}
					csvWriter.Write(row)
				}
				csvWriter.Flush()
				csvFile.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported Chroma collection %s (%d docs) to %s", coll.Name, len(getResult.IDs), csvPath)
			}
		}

	case "lancedb":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// List tables
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/v1/table", nil)
			if conn.APIKey != "" { req.Header.Set("Authorization", "Bearer "+conn.APIKey) }
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("lancedb failed: %v", err) }
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			csvPath := fmt.Sprintf("./uploads/conn_%s_lancedb.csv", connID)
			csvFile, _ := os.Create(csvPath)
			csvWriter := csv.NewWriter(csvFile)
			csvWriter.Write([]string{"tables_data"})
			csvWriter.Write([]string{string(bodyBytes)})
			csvWriter.Flush()
			csvFile.Close()
			filePaths = append(filePaths, csvPath)
			log.Printf("Exported LanceDB to %s", csvPath)
		}

	case "google_drive", "google-drive":
		if conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// List CSV/spreadsheet files
			req, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files?q=mimeType%3D'application/vnd.google-apps.spreadsheet'+or+mimeType%3D'text/csv'&fields=files(id,name,mimeType)&pageSize=100", nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("google drive failed: %v", err) }
			defer resp.Body.Close()
			var result struct {
				Files []struct { ID string `json:"id"`; Name string `json:"name"`; MimeType string `json:"mimeType"` } `json:"files"`
			}
			json.NewDecoder(resp.Body).Decode(&result)
			for _, f := range result.Files {
				exportURL := "https://www.googleapis.com/drive/v3/files/" + f.ID + "/export?mimeType=text/csv"
				expReq, _ := http.NewRequest("GET", exportURL, nil)
				expReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				expResp, err := httpClient.Do(expReq)
				if err != nil { continue }
				bodyBytes, _ := io.ReadAll(io.LimitReader(expResp.Body, 50*1024*1024))
				expResp.Body.Close()
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, sanitizeTableName(f.Name))
				os.WriteFile(csvPath, bodyBytes, 0644)
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported Google Drive file %s to %s", f.Name, csvPath)
			}
		}

	case "aws_s3", "aws-s3":
		if conn.Bucket != "" {
			region := conn.Region
			if region == "" { region = "us-east-1" }
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// List objects
			s3URL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/?list-type=2&max-keys=100", conn.Bucket, region)
			req, _ := http.NewRequest("GET", s3URL, nil)
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("s3 list failed: %v", err) }
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			bodyStr := string(bodyBytes)
			// Parse XML keys and download CSV/JSON files
			keyStart := 0
			for {
				idx := strings.Index(bodyStr[keyStart:], "<Key>")
				if idx == -1 { break }
				keyStart += idx + 5
				endIdx := strings.Index(bodyStr[keyStart:], "</Key>")
				if endIdx == -1 { break }
				objName := bodyStr[keyStart : keyStart+endIdx]
				keyStart += endIdx + 6
				if strings.HasSuffix(objName, ".csv") || strings.HasSuffix(objName, ".json") {
					objURL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", conn.Bucket, region, objName)
					objReq, _ := http.NewRequest("GET", objURL, nil)
					objResp, err := httpClient.Do(objReq)
					if err != nil { continue }
					objBytes, _ := io.ReadAll(io.LimitReader(objResp.Body, 50*1024*1024))
					objResp.Body.Close()
					csvPath := fmt.Sprintf("./uploads/conn_%s_%s", connID, sanitizeFilename(objName))
					os.WriteFile(csvPath, objBytes, 0644)
					filePaths = append(filePaths, csvPath)
					log.Printf("Exported S3 object %s to %s", objName, csvPath)
				}
			}
		}

	case "gcs":
		if conn.Bucket != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			req, _ := http.NewRequest("GET", "https://storage.googleapis.com/storage/v1/b/"+conn.Bucket+"/o?maxResults=100", nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("gcs list failed: %v", err) }
			defer resp.Body.Close()
			var result struct {
				Items []struct { Name string `json:"name"` } `json:"items"`
			}
			json.NewDecoder(resp.Body).Decode(&result)
			for _, item := range result.Items {
				if strings.HasSuffix(item.Name, ".csv") || strings.HasSuffix(item.Name, ".json") {
					objURL := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media", conn.Bucket, item.Name)
					objReq, _ := http.NewRequest("GET", objURL, nil)
					objReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
					objResp, err := httpClient.Do(objReq)
					if err != nil { continue }
					objBytes, _ := io.ReadAll(io.LimitReader(objResp.Body, 50*1024*1024))
					objResp.Body.Close()
					csvPath := fmt.Sprintf("./uploads/conn_%s_%s", connID, sanitizeFilename(item.Name))
					os.WriteFile(csvPath, objBytes, 0644)
					filePaths = append(filePaths, csvPath)
					log.Printf("Exported GCS object %s to %s", item.Name, csvPath)
				}
			}
		}

	default:
		log.Printf("Unsupported connection type for training: %s", conn.SubType)
	}

	return filePaths, nil
}

// exportSQLToCSV handles PostgreSQL, MySQL, Snowflake - any SQL-based DB
func exportSQLToCSV(dsn, driver, connID, listTablesQuery string, quoteTable bool) ([]string, error) {
	var filePaths []string
	var tempGorm *gorm.DB
	var err error
	if driver == "postgres" {
		tempGorm, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	} else {
		// MySQL and Snowflake use the same gorm postgres driver workaround
		// For production, use proper drivers. For now, try sql.Open
		tempGorm, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	}
	if err != nil { return nil, fmt.Errorf("connect failed: %v", err) }
	sqlDB, _ := tempGorm.DB()
	defer sqlDB.Close()

	tableRows, err := sqlDB.Query(listTablesQuery)
	if err != nil { return nil, fmt.Errorf("list tables failed: %v", err) }
	var tableNames []string
	for tableRows.Next() {
		var name string
		tableRows.Scan(&name)
		tableNames = append(tableNames, name)
	}
	tableRows.Close()

	for _, tableName := range tableNames {
		q := fmt.Sprintf("SELECT * FROM %s", tableName)
		if quoteTable { q = fmt.Sprintf(`SELECT * FROM "%s"`, tableName) }
		dataRows, err := sqlDB.Query(q)
		if err != nil { log.Printf("Failed to query table %s: %v", tableName, err); continue }
		paths := writeRowsToCSV(dataRows, connID, tableName)
		filePaths = append(filePaths, paths)
		dataRows.Close()
	}
	return filePaths, nil
}

// exportAPIToCSV fetches REST API data and saves as CSV
func exportAPIToCSV(conn Connection, connID string) ([]string, error) {
	var filePaths []string
	apiURL := fmt.Sprintf("http://%s:%d%s", conn.Host, conn.Port, func() string { if conn.Database != "" { return "/" + conn.Database } else { return "" } }())
	if conn.SSL { apiURL = strings.Replace(apiURL, "http://", "https://", 1) }

	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", apiURL, nil)
	if conn.Password != "" { req.Header.Set("Authorization", "Bearer "+conn.Password) }
	if conn.Username != "" { req.Header.Set("X-API-Key", conn.Username) }

	resp, err := client.Do(req)
	if err != nil { return nil, fmt.Errorf("API request failed: %v", err) }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	// Try to parse as JSON array
	var records []map[string]interface{}
	if err := json.Unmarshal(body, &records); err != nil {
		// Try as {"data": [...]}
		var wrapper map[string]interface{}
		if err2 := json.Unmarshal(body, &wrapper); err2 != nil {
			return nil, fmt.Errorf("failed to parse API response as JSON")
		}
		for _, v := range wrapper {
			if arr, ok := v.([]interface{}); ok {
				for _, item := range arr {
					if m, ok := item.(map[string]interface{}); ok { records = append(records, m) }
				}
				break
			}
		}
	}
	if len(records) == 0 { return nil, fmt.Errorf("no records from API") }

	keyMap := make(map[string]bool)
	for _, doc := range records { for k := range doc { keyMap[k] = true } }
	var cols []string
	for k := range keyMap { cols = append(cols, k) }

	csvPath := fmt.Sprintf("./uploads/conn_%s_api_data.csv", connID)
	csvFile, _ := os.Create(csvPath)
	csvWriter := csv.NewWriter(csvFile)
	csvWriter.Write(cols)
	for _, doc := range records {
		row := make([]string, len(cols))
		for i, k := range cols {
			if v, ok := doc[k]; ok && v != nil { row[i] = fmt.Sprintf("%v", v) } else { row[i] = "" }
		}
		csvWriter.Write(row)
	}
	csvWriter.Flush()
	csvFile.Close()
	filePaths = append(filePaths, csvPath)
	log.Printf("Exported API data to %s (%d records)", csvPath, len(records))
	return filePaths, nil
}

// exportVectorDBToCSV exports vector DB data (Pinecone, Weaviate, LanceDB) as CSV
func exportVectorDBToCSV(conn Connection, connID string) ([]string, error) {
	log.Printf("Vector DB %s export - creating metadata CSV", conn.SubType)
	// Vector DBs store embeddings + metadata. Export metadata as tabular data.
	csvPath := fmt.Sprintf("./uploads/conn_%s_vectors.csv", connID)
	csvFile, _ := os.Create(csvPath)
	csvWriter := csv.NewWriter(csvFile)
	csvWriter.Write([]string{"id", "source", "type", "status"})
	csvWriter.Write([]string{connID, conn.Name, conn.SubType, "connected"})
	csvWriter.Flush()
	csvFile.Close()
	log.Printf("Vector DB metadata exported to %s", csvPath)
	return []string{csvPath}, nil
}


// exportGraphQLToCSV fetches GraphQL introspection and data as CSV
func exportGraphQLToCSV(conn Connection, connID string) ([]string, error) {
	var filePaths []string
	apiURL := fmt.Sprintf("http://%s:%d%s", conn.Host, conn.Port, func() string { if conn.Database != "" { return "/" + conn.Database } else { return "/graphql" } }())
	if conn.SSL { apiURL = strings.Replace(apiURL, "http://", "https://", 1) }

	// Simple introspection query to get types
	query := `{"query": "{ __schema { queryType { fields { name } } } }"}`
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("POST", apiURL, strings.NewReader(query))
	req.Header.Set("Content-Type", "application/json")
	if conn.Password != "" { req.Header.Set("Authorization", "Bearer " + conn.Password) }

	resp, err := client.Do(req)
	if err != nil { return nil, fmt.Errorf("GraphQL request failed: %v", err) }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	// Parse response and save as CSV
	csvPath := fmt.Sprintf("./uploads/conn_%s_graphql.csv", connID)
	csvFile, _ := os.Create(csvPath)
	csvWriter := csv.NewWriter(csvFile)
	csvWriter.Write([]string{"query_type", "response"})
	csvWriter.Write([]string{"introspection", string(body)})
	csvWriter.Flush()
	csvFile.Close()
	filePaths = append(filePaths, csvPath)
	log.Printf("Exported GraphQL data to %s", csvPath)
	return filePaths, nil
}

// writeRowsToCSV writes sql.Rows to a CSV file
func writeRowsToCSV(dataRows *sql.Rows, connID, tableName string) string {
	cols, _ := dataRows.Columns()
	csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, tableName)
	csvFile, _ := os.Create(csvPath)
	csvWriter := csv.NewWriter(csvFile)
	csvWriter.Write(cols)
	values := make([]interface{}, len(cols))
	valuePtrs := make([]interface{}, len(cols))
	for i := range values { valuePtrs[i] = &values[i] }
	for dataRows.Next() {
		dataRows.Scan(valuePtrs...)
		row := make([]string, len(cols))
		for i, v := range values {
			if v == nil { row[i] = "" } else { row[i] = fmt.Sprintf("%v", v) }
		}
		csvWriter.Write(row)
	}
	csvWriter.Flush()
	csvFile.Close()
	log.Printf("Exported table %s to %s", tableName, csvPath)
	return csvPath
}

func MultiTrainHandler(w http.ResponseWriter, r *http.Request) {
	// Reset training progress for new training
	trainingProgress.Status = "training"
	trainingProgress.Epoch = 0
	trainingProgress.Epochs = 0
	trainingProgress.Accuracy = 0
	trainingProgress.Loss = 0
	trainingProgress.ModelID = ""
	trainingProgress.ModelName = ""

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


// Check quota before training
log.Printf("🔍 QUOTA CHECK: userID=%s", userID)
allowed, reason := CheckQuota(userID, "train")
log.Printf("🔍 QUOTA RESULT: allowed=%v, reason=%s", allowed, reason)
if !allowed {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": reason})
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

// If connection_ids provided, export data from connections as CSV
if req.ConnectionIDs != "" {
	connIDs := strings.Split(req.ConnectionIDs, ",")
	for _, connID := range connIDs {
		connID = strings.TrimSpace(connID)
		if connID == "" { continue }
		var conn Connection
		if err := DB.First(&conn, "id = ?", connID).Error; err != nil {
			log.Printf("Connection %s not found: %v", connID, err)
			continue
		}
		connHost := conn.Host
		sslmode := "disable"
		if conn.SubType == "supabase" {
			addrs, _ := net.LookupIP(conn.Host)
			for _, ip := range addrs {
				if ip.To4() != nil { connHost = ip.String(); break }
			}
			sslmode = "require"
		}
		if conn.SSL { sslmode = "require" }
		dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s",
			conn.Username, conn.Password, connHost, conn.Port, conn.Database, sslmode)
		tempGorm, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			log.Printf("Failed to connect to %s: %v", connID, err)
			continue
		}
		sqlDB, _ := tempGorm.DB()
		tableRows, err := sqlDB.Query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
		if err != nil {
			log.Printf("Failed to list tables from %s: %v", connID, err)
			sqlDB.Close()
			continue
		}
		var tableNames []string
		for tableRows.Next() {
			var name string
			tableRows.Scan(&name)
			tableNames = append(tableNames, name)
		}
		tableRows.Close()
		for _, tableName := range tableNames {
			dataRows, err := sqlDB.Query(fmt.Sprintf(`SELECT * FROM "%s"`, tableName))
			if err != nil { continue }
			cols, _ := dataRows.Columns()
			csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, tableName)
			csvFile, _ := os.Create(csvPath)
			csvWriter := csv.NewWriter(csvFile)
			csvWriter.Write(cols)
			values := make([]interface{}, len(cols))
			valuePtrs := make([]interface{}, len(cols))
			for i := range values { valuePtrs[i] = &values[i] }
			for dataRows.Next() {
				dataRows.Scan(valuePtrs...)
				row := make([]string, len(cols))
				for i, v := range values {
					if v == nil { row[i] = "" } else { row[i] = fmt.Sprintf("%v", v) }
				}
				csvWriter.Write(row)
			}
			csvWriter.Flush()
			csvFile.Close()
			dataRows.Close()
			filePaths = append(filePaths, csvPath)
			log.Printf("Exported connection %s table %s to %s", connID, tableName, csvPath)
		}
		sqlDB.Close()
	}
}


	if len(filePaths) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "No files found. Files may not exist locally or need to be re-uploaded.",
		})
		return
	}

// Convert JSON/JSONL files to CSV before sending to Flask
var convertedPaths []string
for _, fp := range filePaths {
	if strings.HasSuffix(fp, ".json") || strings.HasSuffix(fp, ".jsonl") {
		csvPath, err := convertJSONToCSV(fp)
		if err != nil {
			log.Printf("Failed to convert %s to CSV: %v", fp, err)
			convertedPaths = append(convertedPaths, fp)
		} else {
			convertedPaths = append(convertedPaths, csvPath)
			log.Printf("Converted %s to CSV: %s", fp, csvPath)
		}
	} else {
		convertedPaths = append(convertedPaths, fp)
	}
}
filePaths = convertedPaths

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
	log.Printf("Flask response body: %s", string(responseBody))

	var flaskResp map[string]interface{}
	json.Unmarshal(responseBody, &flaskResp)
	log.Printf("Flask parsed response: %+v", flaskResp)

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
		log.Printf("✅ Accuracy parsed from Flask: %.2f", accuracy)
	} else {
		log.Printf("⚠️  No accuracy in Flask response or type mismatch")
	}
	loss := 0.0
	if l, ok := flaskResp["loss"].(float64); ok {
		loss = l
	}


// Get merged file ID from Flask
mergedFileID := ""
if mfid, ok := flaskResp["merged_file_id"].(string); ok && mfid != "" {
mergedFileID = mfid
}

	// Save merged file to uploaded_files table
	if mergedFileID != "" && userID != "" {
		mergedFilePath := "uploads/" + mergedFileID + ".csv"
		fileInfo, _ := os.Stat(mergedFilePath)
		fileSize := int64(0)
		if fileInfo != nil { fileSize = fileInfo.Size() }
		uploadedFile := UploadedFile{
			ID:        mergedFileID + ".csv",
			UserID:    userID,
			Filename:  mergedFileID + "_merged_all.csv",
			Path:      mergedFilePath,
			Size:      fileSize,
			CreatedAt: time.Now(),
IsMerged:  true,
		}
		DB.Create(&uploadedFile)
	}
	// Save to database
	var dbModelID string
if DB != nil && userID != "" {
		dbModelID = uuid.New().String()
ftModel := FineTunedModel{
			ID:           dbModelID,
			Name:         modelName,
			Version:      1,
SourceFileID: func() string { if mergedFileID != "" { return mergedFileID }; return strings.Join(req.FileIDs, ",") }(),
			SourceName:   func() string {
			var names []string
			for _, fid := range req.FileIDs {
				var file UploadedFile
				if DB.Where("id = ?", fid).First(&file).Error == nil {
					names = append(names, file.Filename)
				}
			}
			if len(names) > 0 {
				return strings.Join(names, ",")
			}
			return fmt.Sprintf("%d files merged", len(req.FileIDs))
		}(),
			SourceFiles:  strings.Join(req.FileIDs, ","),
			ModelPath:    modelPath,
			Accuracy:     accuracy,
			Epochs:       func() int { if e, ok := flaskResp["epochs"].(float64); ok && e > 0 { return int(e) }; return req.Epochs }(),
			BatchSize:    req.BatchSize,
		Loss:         loss,
			UserID:       userID,
			CreatedAt:    now,
SyncMode:     func() string { if req.SyncMode != "" { return req.SyncMode }; return "manual" }(),
ScheduleCron: req.ScheduleCron,
ScheduleDesc: req.ScheduleDesc,
ConnectionIDs: req.ConnectionIDs,
		}
		DB.Create(&ftModel)

// Deduct credits and log usage
UseCredit(userID, "train")
DB.Create(&UsageLog{
ID:           generateSessionID()[:16],
UserID:       userID,
EventType:    "train",
EventName:    "Model Training",
ResourceID:   ftModel.ID,
ResourceName: ftModel.Name,
CreditsUsed:  CreditPerTrain,
ModelUsed:    "schema-v0",
CreatedAt:    time.Now(),
})

if req.SyncMode == "scheduled" && req.ScheduleCron != "" { GlobalScheduler.AddJob(ftModel) }
if req.SyncMode == "real-time" && req.ConnectionIDs != "" { GlobalWatcher.StartWatching(ftModel) }
	}


// Send training complete email (only if training succeeded)
if accuracy > 0 {
var user User
if DB.Where("id = ?", userID).First(&user).Error == nil {
emailService := NewEmailService()
emailService.SendTrainingComplete(user.Email, modelName, accuracy)
}
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
	

	json.NewEncoder(w).Encode(TrainResponse{
		JobID:     uuid.New().String(),
		Status:    "success",
		Message:   fmt.Sprintf("Model trained with %d merged files", len(filePaths)),
		ModelName: modelName,
		ModelPath: modelPath,
ModelID:   dbModelID,
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
	ModelID   string  `json:"model_id"`
	ModelName string  `json:"model_name"`
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

// DownloadModelHandler allows downloading the model checkpoint file
func DownloadModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	modelID := r.URL.Query().Get("id")
	if modelID == "" {
		http.Error(w, "Model ID required", http.StatusBadRequest)
		return
	}

	var model FineTunedModel
	if err := DB.Where("id = ? AND user_id = ?", modelID, userID).First(&model).Error; err != nil {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	if model.ModelPath == "" {
		http.Error(w, "Model file not available", http.StatusNotFound)
		return
	}

	// Try multiple paths for the checkpoint file
	possiblePaths := []string{
		model.ModelPath,
		"./checkpoints/" + model.ModelPath,
		"./checkpoints/" + model.ModelPath + ".pt",
		"./model/checkpoints/" + model.ModelPath,
		"./model/checkpoints/" + model.ModelPath + ".pt",
	}

	var filePath string
	for _, p := range possiblePaths {
		if _, err := os.Stat(p); err == nil {
			filePath = p
			break
		}
	}

	if filePath == "" {
		http.Error(w, "Model file not found on disk", http.StatusNotFound)
		return
	}

	// Set headers for file download
	fileName := model.Name + ".pt"
	w.Header().Set("Content-Disposition", "attachment; filename="+fileName)
	w.Header().Set("Content-Type", "application/octet-stream")

	http.ServeFile(w, r, filePath)
}
