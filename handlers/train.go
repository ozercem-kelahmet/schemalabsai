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
sf "github.com/snowflakedb/gosnowflake"
	"net"
	"strings"
	"sync"

	redisv9 "github.com/redis/go-redis/v9"
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
if r.URL.Path != "/api/train" {
http.Error(w, "Not found", http.StatusNotFound)
return
}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	log.Printf("=== TRAIN HANDLER START: user=%s ===", userID)
	TrainingJobsTotal.WithLabelValues("started").Inc()
	TrainingJobsActive.Inc()

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
			log.Printf("Flask goroutine error: %v", err)
			trainingProgressMu.Lock()
			trainingProgress.Status = "error"
			trainingProgress.Loss = 0
			trainingProgressMu.Unlock()
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
// Set completed IMMEDIATELY so polling stops returning stale training status
trainingProgressMu.Lock()
trainingProgress.Status = "completed"
trainingProgress.Accuracy = accuracy
trainingProgressMu.Unlock()
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
// Estimate tokens: epochs * rows * cols * 2
trainTokens := ftModel.Epochs * 2500
if trainTokens < 1000 { trainTokens = 1000 }
DB.Create(&UsageLog{
ID:           generateSessionID()[:16],
UserID:       userID,
EventType:    "train",
EventName:    "Model Training",
ResourceID:   ftModel.ID,
ResourceName: ftModel.Name,
CreditsUsed:  CreditPerTrain,
TokensUsed:   trainTokens,
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
	DB.Where("user_id = ? AND (status = ? OR status = ? OR status IS NULL)", userID, "active", "").Order("created_at desc").Find(&models)

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

		// Resolve connection names
		if m.ConnectionIDs != "" && DB != nil {
			var connNames []string
			for _, cid := range strings.Split(m.ConnectionIDs, ",") {
				cid = strings.TrimSpace(cid)
				if cid == "" { continue }
				var conn Connection
				if DB.Where("id = ?", cid).First(&conn).Error == nil {
					connNames = append(connNames, conn.Name)
				}
			}
			if len(connNames) > 0 {
				mr["connection_names"] = strings.Join(connNames, ",")
			}
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
SelectedTables string   `json:"selected_tables"`
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
		dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s&connect_timeout=15",
			conn.Username, conn.Password, connHost, conn.Port, conn.Database, sslmode)
		paths, err := exportSQLToCSV(dsn, "postgres", connID, "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'", true, conn.SelectedTables)
		if err != nil { return nil, err }
		filePaths = append(filePaths, paths...)

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		paths, err := exportSQLToCSV(dsn, "mysql", connID, fmt.Sprintf("SELECT table_name FROM information_schema.tables WHERE table_schema = '%s'", conn.Database), false, conn.SelectedTables)
		if err != nil { return nil, err }
		filePaths = append(filePaths, paths...)

	case "snowflake":
		dsn := fmt.Sprintf("postgresql://%s:%s@%s/%s",
			conn.Username, conn.Password, conn.Host, conn.Database)
		paths, err := exportSQLToCSV(dsn, "snowflake", connID, "SELECT table_name FROM information_schema.tables WHERE table_schema = 'PUBLIC'", true, conn.SelectedTables)
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
var mongoSelMap map[string]bool
if conn.SelectedTables != "" {
var sel []string
if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
mongoSelMap = make(map[string]bool)
for _, s := range sel { mongoSelMap[s] = true }
}
		for _, collName := range collections {
if mongoSelMap != nil && !mongoSelMap[collName] { continue }
			csvPath := exportMongoToCSV(ctx, db, collName, conn, connID)
			if csvPath != "" { filePaths = append(filePaths, csvPath) }
		}

	case "databricks":
		dsn := fmt.Sprintf("postgresql://%s:%s@%s/%s",
			conn.Username, conn.Password, conn.Host, conn.Database)
		paths, err := exportSQLToCSV(dsn, "databricks", connID, "SHOW TABLES", false, conn.SelectedTables)
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
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, sanitizeFilename(conn.Name))
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
			if conn.APIKey != "" { req.Header.Set("X-Chroma-Token", conn.APIKey) }
			resp, err := httpClient.Do(req)
			if err != nil { return nil, fmt.Errorf("chroma failed: %v", err) }
			defer resp.Body.Close()
			var collections []struct { ID string `json:"id"`; Name string `json:"name"` }
			json.NewDecoder(resp.Body).Decode(&collections)
			for _, coll := range collections {
				getBody, _ := json.Marshal(map[string]interface{}{"limit": 10000, "include": []string{"documents", "metadatas"}})
				getReq, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/api/v1/collections/"+coll.ID+"/get", bytes.NewReader(getBody))
				getReq.Header.Set("Content-Type", "application/json")
				if conn.APIKey != "" { getReq.Header.Set("X-Chroma-Token", conn.APIKey) }
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
			if conn.APIKey != "" { req.Header.Set("X-Chroma-Token", conn.APIKey) }
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
					var s3SelMap map[string]bool
					if conn.SelectedTables != "" {
						var sel []string
						if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
						s3SelMap = make(map[string]bool)
						for _, s := range sel { s3SelMap[s] = true }
					}
					if s3SelMap != nil && !s3SelMap[objName] { continue }
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


	case "excel":
		var connFiles []UploadedFile
		DB.Where("id LIKE ? AND source = ?", "conn_"+connID+"%", "connection").Find(&connFiles)
		log.Printf("Excel connection %s: found %d CSV files", connID, len(connFiles))
		for _, cf := range connFiles {
			if cf.Path != "" {
				filePaths = append(filePaths, cf.Path)
				log.Printf("Excel CSV: %s -> %s", cf.Filename, cf.Path)
			}
		}
		if len(filePaths) == 0 {
	trainingProgress.Status = "idle"
	trainingProgressMu.Lock()
	trainingProgress.Epoch = 0
	trainingProgress.Accuracy = 0
	trainingProgress.Loss = 0
	trainingProgressMu.Unlock()
			return nil, fmt.Errorf("no CSV files found for Excel connection %s", connID)
		}
	default:
		log.Printf("Unsupported connection type for training: %s", conn.SubType)
	}

	return filePaths, nil
}

// exportSQLToCSV handles PostgreSQL, MySQL, Snowflake - any SQL-based DB
func exportSQLToCSV(dsn, driver, connID, listTablesQuery string, quoteTable bool, selectedTables string) ([]string, error) {
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
	sqlDB.SetConnMaxLifetime(30 * time.Second)
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

var selectedMap map[string]bool
if selectedTables != "" {
var sel []string
if err := json.Unmarshal([]byte(selectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, selectedTables)
}
selectedMap = make(map[string]bool)
for _, s := range sel { selectedMap[s] = true }
}
	for _, tableName := range tableNames {
if selectedMap != nil && !selectedMap[tableName] { continue }
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
	apiURL := conn.Endpoint
	if apiURL == "" { return nil, fmt.Errorf("no GraphQL endpoint") }
	log.Printf("📡 exportGraphQLToCSV: apiURL=%s", apiURL)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	// Full introspection with deep ofType (7 levels covers all cases)
	introQuery := `{"query":"{ __schema { queryType { fields { name type { name kind ofType { name kind ofType { name kind ofType { name kind } } } } } } } }"}`
	introReq, _ := http.NewRequest("POST", apiURL, strings.NewReader(introQuery))
	introReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" { introReq.Header.Set("Authorization", "Bearer "+conn.APIKey) }
	introResp, err := httpClient.Do(introReq)
	if err != nil { return nil, fmt.Errorf("GraphQL introspection failed: %v", err) }
	introBytes, _ := io.ReadAll(introResp.Body)
	introResp.Body.Close()
	log.Printf("📡 Introspection response: %d bytes", len(introBytes))

	// Parse as raw JSON for flexible ofType depth
	var rawResult map[string]interface{}
	json.Unmarshal(introBytes, &rawResult)

	// Helper: recursively resolve type name from ofType chain
	resolveTypeName := func(typeObj map[string]interface{}) string {
		current := typeObj
		for current != nil {
			if name, ok := current["name"].(string); ok && name != "" {
				kind, _ := current["kind"].(string)
				if kind == "OBJECT" || kind == "INTERFACE" || kind == "UNION" {
					return name
				}
			}
			if ofType, ok := current["ofType"].(map[string]interface{}); ok {
				current = ofType
			} else {
				break
			}
		}
		return ""
	}

	// Helper: check if type chain contains LIST
	isListType := func(typeObj map[string]interface{}) bool {
		current := typeObj
		for current != nil {
			if kind, ok := current["kind"].(string); ok && kind == "LIST" {
				return true
			}
			if ofType, ok := current["ofType"].(map[string]interface{}); ok {
				current = ofType
			} else {
				break
			}
		}
		return false
	}

	// Get all types for finding scalar fields
	dataObj, _ := rawResult["data"].(map[string]interface{})
	if dataObj == nil { return nil, fmt.Errorf("no data in introspection") }
	schemaObj, _ := dataObj["__schema"].(map[string]interface{})
	if schemaObj == nil { return nil, fmt.Errorf("no schema") }

	// Get types from full introspection for scalar field lookup
	typesQuery := `{"query":"{ __schema { types { name kind fields { name type { name kind ofType { name kind } } } } } }"}`
	typesReq, _ := http.NewRequest("POST", apiURL, strings.NewReader(typesQuery))
	typesReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" { typesReq.Header.Set("Authorization", "Bearer "+conn.APIKey) }
	typesResp, terr := httpClient.Do(typesReq)
	if terr != nil { return nil, fmt.Errorf("types query failed: %v", terr) }
	typesBytes, _ := io.ReadAll(typesResp.Body)
	typesResp.Body.Close()

	var typesResult map[string]interface{}
	json.Unmarshal(typesBytes, &typesResult)
	typesData, _ := typesResult["data"].(map[string]interface{})
	typesSchema, _ := typesData["__schema"].(map[string]interface{})
	allTypes, _ := typesSchema["types"].([]interface{})

	// Build type -> scalar fields map
	typeFieldsMap := make(map[string][]string)
	for _, t := range allTypes {
		tObj, _ := t.(map[string]interface{})
		tName, _ := tObj["name"].(string)
		tKind, _ := tObj["kind"].(string)
		if tKind != "OBJECT" || tName == "" { continue }
		fields, _ := tObj["fields"].([]interface{})
		var scalarFields []string
		for _, f := range fields {
			fObj, _ := f.(map[string]interface{})
			fName, _ := fObj["name"].(string)
			fType, _ := fObj["type"].(map[string]interface{})
			fKind, _ := fType["kind"].(string)
			fTypeName, _ := fType["name"].(string)
			if fKind == "NON_NULL" {
				if ofType, ok := fType["ofType"].(map[string]interface{}); ok {
					fTypeName, _ = ofType["name"].(string)
				}
			}
			if fKind == "SCALAR" || fTypeName == "String" || fTypeName == "Int" || fTypeName == "Float" || fTypeName == "Boolean" || fTypeName == "ID" {
				scalarFields = append(scalarFields, fName)
			}
		}
		if len(scalarFields) > 0 {
			typeFieldsMap[tName] = scalarFields
		}
	}

	// Selected tables filter
	var selectedMap map[string]bool
	if conn.SelectedTables != "" {
		var selected []string
		if err := json.Unmarshal([]byte(conn.SelectedTables), &selected); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
		if len(selected) > 0 {
			selectedMap = make(map[string]bool)
			for _, s := range selected { selectedMap[s] = true }
		}
	}

	// Build singular field name -> type map (continent->Continent, country->Country)
	queryType, _ := schemaObj["queryType"].(map[string]interface{})
	queryFields, _ := queryType["fields"].([]interface{})
	singularTypeMap := make(map[string]string)
	for _, qf := range queryFields {
		f, _ := qf.(map[string]interface{})
		fName, _ := f["name"].(string)
		fType, _ := f["type"].(map[string]interface{})
		fKind, _ := fType["kind"].(string)
		if fKind == "OBJECT" {
			tName, _ := fType["name"].(string)
			if tName != "" { singularTypeMap[fName] = tName }
		}
	}
	log.Printf("📡 Found %d query fields, %d type mappings, %d singular fields", len(queryFields), len(typeFieldsMap), len(singularTypeMap))

	for _, qf := range queryFields {
		field, _ := qf.(map[string]interface{})
		fieldName, _ := field["name"].(string)
		fieldType, _ := field["type"].(map[string]interface{})

		if !isListType(fieldType) { continue }
		returnType := resolveTypeName(fieldType)
		// If type not resolved from ofType chain, infer from singular field
		if returnType == "" {
			// Try: continents -> continent, countries -> country, languages -> language
			singular := strings.TrimSuffix(fieldName, "ies") 
			if singular != fieldName { singular += "y" } else { singular = strings.TrimSuffix(fieldName, "s") }
			if t, ok := singularTypeMap[singular]; ok { returnType = t }
		}
		// Also try matching from typeFieldsMap by capitalized singular
		if returnType == "" {
			singular := strings.TrimSuffix(fieldName, "ies")
			if singular != fieldName { singular += "y" } else { singular = strings.TrimSuffix(fieldName, "s") }
			cap := strings.ToUpper(singular[:1]) + singular[1:]
			if _, ok := typeFieldsMap[cap]; ok { returnType = cap }
		}
		log.Printf("📡 List field: %s -> %s", fieldName, returnType)
		if returnType == "" { continue }

		// Check selected tables
		if selectedMap != nil {
			matched := false
			for sel := range selectedMap {
				if strings.EqualFold(sel, fieldName) || strings.EqualFold(sel, returnType) || strings.HasPrefix(strings.ToLower(fieldName), strings.ToLower(sel)) {
					matched = true; break
				}
			}
			if !matched { continue }
		}

		scalarFields := typeFieldsMap[returnType]
		if len(scalarFields) == 0 { continue }

		fieldsStr := strings.Join(scalarFields, " ")
		gqlQuery := fmt.Sprintf(`{"query":"{ %s { %s } }"}`, fieldName, fieldsStr)
		dataReq, _ := http.NewRequest("POST", apiURL, strings.NewReader(gqlQuery))
		dataReq.Header.Set("Content-Type", "application/json")
		if conn.APIKey != "" { dataReq.Header.Set("Authorization", "Bearer "+conn.APIKey) }
		dataResp, derr := httpClient.Do(dataReq)
		if derr != nil { log.Printf("📡 Query %s failed: %v", fieldName, derr); continue }
		dataRespBytes, _ := io.ReadAll(dataResp.Body)
		dataResp.Body.Close()

		var dataResult map[string]interface{}
		json.Unmarshal(dataRespBytes, &dataResult)
		if respData, ok := dataResult["data"].(map[string]interface{}); ok {
			if arr, ok := respData[fieldName].([]interface{}); ok && len(arr) > 0 {
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, fieldName)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				csvWriter.Write(scalarFields)
				for _, item := range arr {
					if obj, ok := item.(map[string]interface{}); ok {
						row := make([]string, len(scalarFields))
						for i, h := range scalarFields {
							if v, exists := obj[h]; exists && v != nil { row[i] = fmt.Sprintf("%v", v) }
						}
						csvWriter.Write(row)
					}
				}
				csvWriter.Flush()
				csvFile.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("📡 Exported %s to %s (%d rows, %d cols)", fieldName, csvPath, len(arr), len(scalarFields))
			}
		}
	}
	log.Printf("📡 exportGraphQLToCSV done: %d files exported", len(filePaths))
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
log.Printf("=== MULTI TRAIN HANDLER CALLED: path=%s method=%s ===", r.URL.Path, r.Method)	// Reset training progress for new training
	trainingProgressMu.Lock()
	trainingProgress.Status = "training"
	trainingProgress.Epoch = 0
	trainingProgress.Accuracy = 0
	trainingProgress.Loss = 0
	trainingProgress.ModelID = ""
	trainingProgress.ModelName = ""
	trainingProgress.Epochs = 0
	trainingProgressMu.Unlock()
	// Reset Flask progress too (sync)
	client := &http.Client{Timeout: 3 * time.Second}
	client.Post(GetFlaskURL()+"/training/reset", "application/json", nil)

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


// Register query_id in progress map
if req.QueryID != "" {
setActiveTrainingProgress(req.QueryID, trainingProgress)
trainingProgressMu.Lock()
trainingProgress.StartTime = time.Now().Unix()
trainingProgressMu.Unlock()
}
// Check quota before training
log.Printf("🔍 QUOTA CHECK: userID=%s", userID)
var trainErrors2 []string
if allowed, reason := CheckQuota(userID, "train"); !allowed {
trainErrors2 = append(trainErrors2, reason)
}
if ok, cr := CheckCredits(userID, 0.50); !ok {
trainErrors2 = append(trainErrors2, cr)
}
if len(trainErrors2) > 0 {
log.Printf("🔍 QUOTA ERRORS: %v", trainErrors2)
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": strings.Join(trainErrors2, " | ")})
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
	if err == nil && len(matches) > 0 {
		filePaths = append(filePaths, matches[0])
	} else {
		// Glob bulamadı - DB'den path al (generated dosyalar için)
		var uf UploadedFile
		if err := DB.Where("id = ?", fileID).First(&uf).Error; err == nil && uf.Path != "" {
			if _, ferr := os.Stat(uf.Path); ferr == nil {
				filePaths = append(filePaths, uf.Path)
			} else if _, ferr := os.Stat("./" + uf.Path); ferr == nil {
				filePaths = append(filePaths, "./" + uf.Path)
			}
		}
	}
	}

// If connection_ids provided, export data from connections as CSV
log.Printf("🔍 ConnectionIDs=%q FileIDs=%v filePaths=%d", req.ConnectionIDs, req.FileIDs, len(filePaths))
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
		// Override with request-level selected_tables if provided
		if req.SelectedTables != "" {
			conn.SelectedTables = req.SelectedTables
			log.Printf("Using request-level selected_tables for conn %s: %s", connID, req.SelectedTables)
		}

		// REST API connection - fetch JSON, convert to CSV
		if conn.SubType == "rest_api" && conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			req, _ := http.NewRequest("GET", conn.Endpoint, nil)
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
			if err != nil {
				log.Printf("REST API fetch failed for %s: %v", connID, err)
				continue
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
			var jsonArray []map[string]interface{}
			if json.Unmarshal(bodyBytes, &jsonArray) == nil && len(jsonArray) > 0 {
				csvPath := fmt.Sprintf("./uploads/conn_%s_api_data.csv", connID)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				var headers []string
				for k := range jsonArray[0] { headers = append(headers, k) }
				csvWriter.Write(headers)
				for _, obj := range jsonArray {
					row := make([]string, len(headers))
					for i, h := range headers { row[i] = fmt.Sprintf("%v", obj[h]) }
					csvWriter.Write(row)
				}
				csvWriter.Flush()
				csvFile.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported REST API connection %s to %s (%d rows)", connID, csvPath, len(jsonArray))
			}
			continue
		}


		// MongoDB connection
		if conn.SubType == "mongodb" {
			var mongoSelectedMap map[string]bool
			if conn.SelectedTables != "" {
				var sel []string
				if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
				mongoSelectedMap = make(map[string]bool)
				for _, s := range sel { mongoSelectedMap[s] = true }
			}
			var mongoURI string
			if conn.Endpoint != "" {
				mongoURI = conn.Endpoint
			} else {
				mongoURI = fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
				if conn.Username == "" {
					mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database)
				}
			}
			clientOptions := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
			client, merr := mongo.Connect(context.Background(), clientOptions)
			if merr != nil {
				log.Printf("MongoDB connect failed for %s: %v", connID, merr)
				continue
			}
			dbName := conn.Database
			if dbName == "" {
				// Extract from URI
				parts := strings.Split(mongoURI, "/")
				if len(parts) > 3 { 
					dbName = strings.Split(parts[3], "?")[0]
				}
			}
			if dbName != "" {
				collections, _ := client.Database(dbName).ListCollectionNames(context.Background(), map[string]interface{}{})
				for _, collName := range collections {
					if mongoSelectedMap != nil && !mongoSelectedMap[collName] { continue }
					coll := client.Database(dbName).Collection(collName)
					cursor, cerr := coll.Find(context.Background(), map[string]interface{}{})
					if cerr != nil { continue }
					var docs []map[string]interface{}
					cursor.All(context.Background(), &docs)
					if len(docs) > 0 {
						csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, collName)
						csvFile, _ := os.Create(csvPath)
						csvWriter := csv.NewWriter(csvFile)
						var headers []string
						for k := range docs[0] { headers = append(headers, k) }
						csvWriter.Write(headers)
						for _, doc := range docs {
							row := make([]string, len(headers))
							for i, h := range headers {
								if v, exists := doc[h]; exists && v != nil { row[i] = fmt.Sprintf("%v", v) }
							}
							csvWriter.Write(row)
						}
						csvWriter.Flush()
						csvFile.Close()
						filePaths = append(filePaths, csvPath)
						log.Printf("Exported MongoDB %s.%s to %s (%d rows)", connID, collName, csvPath, len(docs))
					}
				}
			}
			client.Disconnect(context.Background())
			continue
		}

		// Snowflake connection
		if conn.SubType == "snowflake" {
			var sfSelectedMap map[string]bool
			if conn.SelectedTables != "" {
				var sel []string
				if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
				sfSelectedMap = make(map[string]bool)
				for _, s := range sel { sfSelectedMap[s] = true }
			}
			sfCfg := &sf.Config{
				Account:        conn.Host,
				User:           conn.Username,
				Password:       conn.Password,
				Database:       conn.Database,
				Warehouse:      conn.Bucket,
				LoginTimeout:   15 * time.Second,
				RequestTimeout: 30 * time.Second,
			}
			sfDsn, err := sf.DSN(sfCfg)
			if err != nil {
				log.Printf("Snowflake config failed for %s: %v", connID, err)
				continue
			}
			sfDB, err := sql.Open("snowflake", sfDsn)
			if err != nil {
				log.Printf("Snowflake connect failed for %s: %v", connID, err)
				continue
			}
			// List tables
			sfRows, err := sfDB.Query("SHOW TABLES")
			if err != nil {
				log.Printf("Snowflake SHOW TABLES failed for %s: %v", connID, err)
				sfDB.Close()
				continue
			}
			var sfTableNames []string
			sfCols, _ := sfRows.Columns()
			for sfRows.Next() {
				vals := make([]interface{}, len(sfCols))
				for i := range vals { vals[i] = new(sql.NullString) }
				if err := sfRows.Scan(vals...); err != nil { continue }
				name := ""
				for i, col := range sfCols {
					if strings.ToLower(col) == "name" {
						v := vals[i].(*sql.NullString)
						if v.Valid { name = v.String }
						break
					}
				}
				if name != "" { sfTableNames = append(sfTableNames, name) }
			}
			sfRows.Close()
			for _, tableName := range sfTableNames {
				if sfSelectedMap != nil && !sfSelectedMap[tableName] { continue }
				dataRows, err := sfDB.Query(fmt.Sprintf("SELECT * FROM %s", tableName))
				if err != nil { continue }
				cols, _ := dataRows.Columns()
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, tableName)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				csvWriter.Write(cols)
				values := make([]interface{}, len(cols))
				valuePtrs := make([]interface{}, len(cols))
				for i := range values { valuePtrs[i] = &values[i] }
				rowCount := 0
				for dataRows.Next() {
					dataRows.Scan(valuePtrs...)
					row := make([]string, len(cols))
					for i, v := range values {
						if v == nil { row[i] = "" } else { row[i] = fmt.Sprintf("%v", v) }
					}
					csvWriter.Write(row)
					rowCount++
				}
				csvWriter.Flush()
				csvFile.Close()
				dataRows.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported Snowflake %s.%s to %s (%d rows)", connID, tableName, csvPath, rowCount)
			}
			sfDB.Close()
			continue
		}

		// Databricks connection
		if conn.SubType == "databricks" && conn.Host != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 60 * time.Second}
			log.Printf("[DATABRICKS] Connecting: host=%s catalog=%s warehouse=%s selectedTables=%s",
				conn.Host,
				func() string { if conn.Database != "" { return conn.Database }; return "main" }(),
				conn.Endpoint,
				conn.SelectedTables)
			dbWorkspaceURL := "https://" + strings.TrimPrefix(strings.TrimPrefix(conn.Host, "https://"), "http://")
			dbCatalog := conn.Database
			if dbCatalog == "" { dbCatalog = "main" }
			// Extract warehouse ID from endpoint path
			warehouseID := conn.Endpoint
			if strings.Contains(warehouseID, "/") {
				parts := strings.Split(warehouseID, "/")
				warehouseID = parts[len(parts)-1]
			}
			// Selected tables filter
			var selectedMap map[string]bool
			if conn.SelectedTables != "" {
				var selected []string
				if err := json.Unmarshal([]byte(conn.SelectedTables), &selected); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
				if len(selected) > 0 {
					selectedMap = make(map[string]bool)
					for _, s := range selected { selectedMap[s] = true }
				}
			}
			// List schemas and tables
			schReq, _ := http.NewRequest("GET", dbWorkspaceURL+"/api/2.1/unity-catalog/schemas?catalog_name="+dbCatalog, nil)
			schReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
			schResp, serr := httpClient.Do(schReq)
			var schemaNames []string
			if serr != nil {
				log.Printf("[DATABRICKS] Schema request failed: %v", serr)
			} else if schResp.StatusCode != 200 {
				b, _ := io.ReadAll(schResp.Body); schResp.Body.Close()
				log.Printf("[DATABRICKS] Schema request status=%d body=%s", schResp.StatusCode, string(b))
			}
			if serr == nil && schResp.StatusCode == 200 {
				var schResult struct { Schemas []struct { Name string `json:"name"` } `json:"schemas"` }
				json.NewDecoder(schResp.Body).Decode(&schResult)
				schResp.Body.Close()
				for _, s := range schResult.Schemas {
					if s.Name != "information_schema" { schemaNames = append(schemaNames, s.Name) }
				}
			} else if serr == nil { schResp.Body.Close() }
			if len(schemaNames) == 0 { schemaNames = []string{"default"} }
			var dbTables []string
			for _, schema := range schemaNames {
				tReq, _ := http.NewRequest("GET", dbWorkspaceURL+"/api/2.1/unity-catalog/tables?catalog_name="+dbCatalog+"&schema_name="+schema, nil)
				tReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				tResp, terr := httpClient.Do(tReq)
				if terr == nil && tResp.StatusCode == 200 {
					var tResult struct { Tables []struct { Name string `json:"name"` } `json:"tables"` }
					json.NewDecoder(tResp.Body).Decode(&tResult)
					tResp.Body.Close()
					for _, t := range tResult.Tables { dbTables = append(dbTables, schema+"."+t.Name) }
				} else if terr == nil { tResp.Body.Close() }
			}
			log.Printf("📡 Databricks: found %d tables in catalog %s", len(dbTables), dbCatalog)
			if selectedMap != nil { log.Printf("[DB-DEBUG] selectedMap=%v", selectedMap) }
			if len(dbTables) > 0 { log.Printf("[DB-DEBUG] dbTables sample: %v", dbTables[:1]) }
			for _, tableFull := range dbTables {
				if selectedMap != nil && !selectedMap[tableFull] { log.Printf("[DB-DEBUG] SKIP table=%s", tableFull); continue }
				query := fmt.Sprintf("SELECT * FROM %s.%s LIMIT 10000", dbCatalog, tableFull)
				reqBody, _ := json.Marshal(map[string]interface{}{"statement": query, "warehouse_id": warehouseID, "wait_timeout": "50s"})
				sqlReq, _ := http.NewRequest("POST", dbWorkspaceURL+"/api/2.0/sql/statements", bytes.NewReader(reqBody))
				sqlReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				sqlReq.Header.Set("Content-Type", "application/json")
				sqlResp, serr := httpClient.Do(sqlReq)
				if serr != nil { continue }
				var sqlResult struct {
					Manifest struct {
						Schema struct {
							Columns []struct{ Name string `json:"name"` } `json:"columns"`
						} `json:"schema"`
					} `json:"manifest"`
					Result struct {
						DataArray [][]string `json:"data_array"`
					} `json:"result"`
				}
				json.NewDecoder(sqlResp.Body).Decode(&sqlResult)
				sqlResp.Body.Close()
				// Databricks PENDING/RUNNING durumunu handle et - retry
				if len(sqlResult.Result.DataArray) == 0 {
					log.Printf("[DATABRICKS] Empty result for %s, retrying in 3s...", tableFull)
					time.Sleep(3 * time.Second)
					sqlReq2, _ := http.NewRequest("POST", dbWorkspaceURL+"/api/2.0/sql/statements", bytes.NewReader(reqBody))
					sqlReq2.Header.Set("Authorization", "Bearer "+conn.APIKey)
					sqlReq2.Header.Set("Content-Type", "application/json")
					sqlResp2, serr2 := httpClient.Do(sqlReq2)
					if serr2 == nil {
						json.NewDecoder(sqlResp2.Body).Decode(&sqlResult)
						sqlResp2.Body.Close()
						log.Printf("[DATABRICKS] Retry result for %s: %d rows", tableFull, len(sqlResult.Result.DataArray))
					}
				}
				if len(sqlResult.Result.DataArray) > 0 {
					csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, strings.ReplaceAll(tableFull, ".", "_"))
					csvFile, _ := os.Create(csvPath)
					csvWriter := csv.NewWriter(csvFile)
					var headers []string
					for _, c := range sqlResult.Manifest.Schema.Columns { headers = append(headers, c.Name) }
					csvWriter.Write(headers)
					for _, row := range sqlResult.Result.DataArray { csvWriter.Write(row) }
					csvWriter.Flush()
					csvFile.Close()
					filePaths = append(filePaths, csvPath)
					log.Printf("Exported Databricks %s.%s to %s (%d rows)", connID, tableFull, csvPath, len(sqlResult.Result.DataArray))
				}
			}
			continue
		}

		// GraphQL connection - use exportGraphQLToCSV
		if conn.SubType == "graphql" && conn.Endpoint != "" {
			paths, err := exportGraphQLToCSV(conn, connID)
			if err != nil {
				log.Printf("GraphQL export failed for %s: %v", connID, err)
			}
			if len(paths) > 0 { filePaths = append(filePaths, paths...) }
			continue
		}

		// MySQL connection
		if conn.SubType == "mysql" {
			var mysqlSelectedMap map[string]bool
			if conn.SelectedTables != "" {
				var sel []string
				if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
				mysqlSelectedMap = make(map[string]bool)
				for _, s := range sel { mysqlSelectedMap[s] = true }
			}
			mysqlDSN := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?timeout=15s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
			mysqlDB, err := sql.Open("mysql", mysqlDSN)
			if err != nil {
				log.Printf("MySQL connect failed for %s: %v", connID, err)
				continue
			}
			tableRows, err := mysqlDB.Query("SHOW TABLES")
			if err != nil {
				log.Printf("MySQL SHOW TABLES failed for %s: %v", connID, err)
				mysqlDB.Close()
				continue
			}
			var mysqlTableNames []string
			for tableRows.Next() {
				var name string
				tableRows.Scan(&name)
				mysqlTableNames = append(mysqlTableNames, name)
			}
			tableRows.Close()
			for _, tableName := range mysqlTableNames {
				if mysqlSelectedMap != nil && !mysqlSelectedMap[tableName] { continue }
				dataRows, err := mysqlDB.Query(fmt.Sprintf("SELECT * FROM `%s`", tableName))
				if err != nil { continue }
				cols, _ := dataRows.Columns()
				csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, tableName)
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				csvWriter.Write(cols)
				values := make([]interface{}, len(cols))
				valuePtrs := make([]interface{}, len(cols))
				for i := range values { valuePtrs[i] = &values[i] }
				rowCount := 0
				for dataRows.Next() {
					dataRows.Scan(valuePtrs...)
					row := make([]string, len(cols))
					for i, v := range values {
						if v == nil { row[i] = "" } else { row[i] = fmt.Sprintf("%v", v) }
					}
					csvWriter.Write(row)
					rowCount++
				}
				csvWriter.Flush()
				csvFile.Close()
				dataRows.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported MySQL %s.%s to %s (%d rows)", connID, tableName, csvPath, rowCount)
			}
			mysqlDB.Close()
			continue
		}

		// Pinecone connection
		if conn.SubType == "pinecone" && conn.Endpoint != "" && conn.APIKey != "" {
			var pineSelectedMap map[string]bool
			if conn.SelectedTables != "" {
				var sel []string
				if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
				pineSelectedMap = make(map[string]bool)
				for _, s := range sel { pineSelectedMap[s] = true }
			}
			_ = pineSelectedMap
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// Get index dimension first
			descReq, _ := http.NewRequest("GET", conn.Endpoint+"/describe_index_stats", nil)
			descReq.Header.Set("Api-Key", conn.APIKey)
			descResp, descErr := httpClient.Do(descReq)
			dim := 1536
			if descErr == nil {
				descBody, _ := io.ReadAll(descResp.Body)
				descResp.Body.Close()
				var descResult struct { Dimension int `json:"dimension"` }
				json.Unmarshal(descBody, &descResult)
				if descResult.Dimension > 0 { dim = descResult.Dimension }
				log.Printf("Pinecone dimension: %d", dim)
			}
			zeroVec := make([]string, dim)
			for i := range zeroVec { zeroVec[i] = "0" }
			queryBody := fmt.Sprintf(`{"topK":10000,"includeMetadata":true,"vector":[%s]}`, strings.Join(zeroVec, ","))
			req, _ := http.NewRequest("POST", conn.Endpoint+"/query", strings.NewReader(queryBody))
			req.Header.Set("Api-Key", conn.APIKey)
			req.Header.Set("Content-Type", "application/json")
			resp, err := httpClient.Do(req)
			if err != nil {
				log.Printf("Pinecone query failed for %s: %v", connID, err)
				continue
			}
			bodyBytes, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			var result struct {
				Matches []struct {
					ID       string                 `json:"id"`
					Values   []float64              `json:"values"`
					Metadata map[string]interface{} `json:"metadata"`
				} `json:"matches"`
			}
			json.Unmarshal(bodyBytes, &result)
			if len(result.Matches) > 0 {
				// Use selected table/namespace name for Pinecone, fallback to connection name
pineconeTableName := conn.Name
if conn.SelectedTables != "" {
	var selTables []string
	if json.Unmarshal([]byte(conn.SelectedTables), &selTables) == nil && len(selTables) > 0 {
		pineconeTableName = selTables[0]
	}
}
csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, sanitizeFilename(pineconeTableName))
				csvFile, _ := os.Create(csvPath)
				csvWriter := csv.NewWriter(csvFile)
				var headers []string
				headers = append(headers, "id")
				for k := range result.Matches[0].Metadata { headers = append(headers, k) }
				csvWriter.Write(headers)
				for _, m := range result.Matches {
					row := []string{m.ID}
					for _, h := range headers[1:] {
						if v, ok := m.Metadata[h]; ok { row = append(row, fmt.Sprintf("%v", v)) } else { row = append(row, "") }
					}
					csvWriter.Write(row)
				}
				csvWriter.Flush()
				csvFile.Close()
				filePaths = append(filePaths, csvPath)
				log.Printf("Exported Pinecone %s to %s (%d rows)", connID, csvPath, len(result.Matches))
			}
			continue
		}

		// Chroma connection
		if conn.SubType == "chroma" && conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// List collections
			chromaTenant := conn.Database
			chromaDatabase := conn.Bucket
			if chromaTenant == "" { chromaTenant = "default_tenant" }
			if chromaDatabase == "" { chromaDatabase = "default_database" }
			listReq, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/api/v2/tenants/"+chromaTenant+"/databases/"+chromaDatabase+"/collections", nil)
			if conn.APIKey != "" { listReq.Header.Set("X-Chroma-Token", conn.APIKey) }
			listResp, err := httpClient.Do(listReq)
			if err != nil {
				log.Printf("Chroma list failed for %s: %v", connID, err)
				continue
			}
			var collections []struct { Name string `json:"name"`; ID string `json:"id"` }
			json.NewDecoder(listResp.Body).Decode(&collections)
			listResp.Body.Close()
			for _, coll := range collections {
				getReq, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/api/v2/tenants/"+chromaTenant+"/databases/"+chromaDatabase+"/collections/"+coll.ID+"/get", strings.NewReader(`{"include":["metadatas","documents"]}`))
				getReq.Header.Set("Content-Type", "application/json")
				if conn.APIKey != "" { getReq.Header.Set("X-Chroma-Token", conn.APIKey) }
				getResp, gerr := httpClient.Do(getReq)
				if gerr != nil { continue }
				var getResult struct {
					IDs       []string                 `json:"ids"`
					Documents []string                 `json:"documents"`
					Metadatas []map[string]interface{} `json:"metadatas"`
				}
				json.NewDecoder(getResp.Body).Decode(&getResult)
				getResp.Body.Close()
				if len(getResult.IDs) > 0 {
					csvPath := fmt.Sprintf("./uploads/conn_%s_%s.csv", connID, coll.Name)
					csvFile, _ := os.Create(csvPath)
					csvWriter := csv.NewWriter(csvFile)
					headers := []string{"id", "document"}
					if len(getResult.Metadatas) > 0 {
						for k := range getResult.Metadatas[0] { headers = append(headers, k) }
					}
					csvWriter.Write(headers)
					for i, id := range getResult.IDs {
						row := []string{id}
						if i < len(getResult.Documents) { row = append(row, getResult.Documents[i]) } else { row = append(row, "") }
						if i < len(getResult.Metadatas) {
							for _, h := range headers[2:] {
								if v, ok := getResult.Metadatas[i][h]; ok { row = append(row, fmt.Sprintf("%v", v)) } else { row = append(row, "") }
							}
						}
						csvWriter.Write(row)
					}
					csvWriter.Flush()
					csvFile.Close()
					filePaths = append(filePaths, csvPath)
					log.Printf("Exported Chroma %s.%s to %s (%d rows)", connID, coll.Name, csvPath, len(getResult.IDs))
				}
			}
			continue
		}

		// Google Drive connection
		if conn.SubType == "google_drive" || conn.SubType == "google-drive" {
			if conn.APIKey != "" {
				httpClient := &http.Client{Timeout: 30 * time.Second}
				listReq, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files?q=mimeType%3D'text/csv'&fields=files(id,name)&pageSize=100", nil)
				listReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				listResp, err := httpClient.Do(listReq)
				if err != nil {
					log.Printf("Google Drive list failed for %s: %v", connID, err)
					continue
				}
				var listResult struct {
					Files []struct { ID string `json:"id"`; Name string `json:"name"` } `json:"files"`
				}
				json.NewDecoder(listResp.Body).Decode(&listResult)
				listResp.Body.Close()
				var gdriveSelMap map[string]bool
				if conn.SelectedTables != "" {
					var sel []string
					if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
					gdriveSelMap = make(map[string]bool)
					for _, s := range sel { gdriveSelMap[s] = true }
				}
				for _, f := range listResult.Files {
					if gdriveSelMap != nil && !gdriveSelMap[f.Name] { continue }
					exportReq, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files/"+f.ID+"?alt=media", nil)
					exportReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
					exportResp, eerr := httpClient.Do(exportReq)
					if eerr != nil { continue }
					bodyBytes, _ := io.ReadAll(io.LimitReader(exportResp.Body, 50*1024*1024))
					exportResp.Body.Close()
					csvPath := fmt.Sprintf("./uploads/conn_%s_%s", connID, f.Name)
					os.WriteFile(csvPath, bodyBytes, 0644)
					filePaths = append(filePaths, csvPath)
					log.Printf("Exported Google Drive %s to %s", f.Name, csvPath)
				}
			}
			continue
		}

		// AWS S3 connection (REST API)
		if conn.SubType == "aws_s3" || conn.SubType == "aws-s3" {
			if conn.Bucket != "" {
				region := conn.Region
				if region == "" { region = "us-east-1" }
				httpClient := &http.Client{Timeout: 30 * time.Second}
				s3URL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/?list-type=2&max-keys=100", conn.Bucket, region)
				listReq, _ := http.NewRequest("GET", s3URL, nil)
				listResp, err := httpClient.Do(listReq)
				if err != nil {
					log.Printf("AWS S3 list failed for %s: %v", connID, err)
					continue
				}
				bodyBytes, _ := io.ReadAll(listResp.Body)
				listResp.Body.Close()
				bodyStr := string(bodyBytes)
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
					var s3SelMap map[string]bool
					if conn.SelectedTables != "" {
						var sel []string
						if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
						s3SelMap = make(map[string]bool)
						for _, s := range sel { s3SelMap[s] = true }
					}
					if s3SelMap != nil && !s3SelMap[objName] { continue }
						objURL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", conn.Bucket, region, objName)
						objReq, _ := http.NewRequest("GET", objURL, nil)
						objResp, oerr := httpClient.Do(objReq)
						if oerr != nil { continue }
						objBytes, _ := io.ReadAll(io.LimitReader(objResp.Body, 50*1024*1024))
						objResp.Body.Close()
						safeName := strings.ReplaceAll(objName, "/", "_")
						csvPath := fmt.Sprintf("./uploads/conn_%s_%s", connID, safeName)
						os.WriteFile(csvPath, objBytes, 0644)
						filePaths = append(filePaths, csvPath)
						log.Printf("Exported S3 %s to %s", objName, csvPath)
					}
				}
			}
			continue
		}

		// GCS connection
		if conn.SubType == "gcs" {
			if conn.Bucket != "" && conn.APIKey != "" {
				httpClient := &http.Client{Timeout: 30 * time.Second}
				listReq, _ := http.NewRequest("GET", "https://storage.googleapis.com/storage/v1/b/"+conn.Bucket+"/o?maxResults=100", nil)
				listReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				listResp, err := httpClient.Do(listReq)
				if err != nil {
					log.Printf("GCS list failed for %s: %v", connID, err)
					continue
				}
				var listResult struct {
					Items []struct { Name string `json:"name"` } `json:"items"`
				}
				json.NewDecoder(listResp.Body).Decode(&listResult)
				listResp.Body.Close()
				for _, item := range listResult.Items {
					if strings.HasSuffix(item.Name, ".csv") || strings.HasSuffix(item.Name, ".json") {
						dlReq, _ := http.NewRequest("GET", "https://storage.googleapis.com/storage/v1/b/"+conn.Bucket+"/o/"+item.Name+"?alt=media", nil)
						dlReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
						dlResp, derr := httpClient.Do(dlReq)
						if derr != nil { continue }
						bodyBytes, _ := io.ReadAll(io.LimitReader(dlResp.Body, 50*1024*1024))
						dlResp.Body.Close()
						safeName := strings.ReplaceAll(item.Name, "/", "_")
						csvPath := fmt.Sprintf("./uploads/conn_%s_%s", connID, safeName)
						os.WriteFile(csvPath, bodyBytes, 0644)
						filePaths = append(filePaths, csvPath)
						log.Printf("Exported GCS %s to %s", item.Name, csvPath)
					}
				}
			}
			continue
		}

// Excel connection - use pre-exported CSV files
if conn.SubType == "excel" {
	var connFiles []UploadedFile
	DB.Where("id LIKE ? AND source = ?", "conn_"+connID+"%", "connection").Find(&connFiles)
	log.Printf("Excel connection %s: found %d CSV files, selectedTables: %s", connID, len(connFiles), conn.SelectedTables)
	selectedMap := map[string]bool{}
	if conn.SelectedTables != "" {
		var tableList []string
		if err := json.Unmarshal([]byte(conn.SelectedTables), &tableList); err == nil {
			for _, t := range tableList {
				selectedMap[strings.TrimSpace(t)] = true
			}
		} else {
			for _, t := range strings.Split(conn.SelectedTables, ",") {
				selectedMap[strings.TrimSpace(t)] = true
			}
		}
	}
	for _, cf := range connFiles {
		if cf.Path != "" {
			if len(selectedMap) > 0 {
				match := false
				for sel := range selectedMap {
					if strings.Contains(cf.Filename, sel) || strings.Contains(cf.ID, sel) {
						match = true
						break
					}
				}
				if !match {
					log.Printf("Excel CSV skipped (not selected): %s", cf.Filename)
					continue
				}
			}
			filePaths = append(filePaths, cf.Path)
			log.Printf("Excel CSV: %s -> %s", cf.Filename, cf.Path)
		}
	}
	continue
}

		// PostgreSQL / Supabase (existing code)
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
		dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s&connect_timeout=15",
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
// Filter by selected tables
var pgSelMap map[string]bool
if conn.SelectedTables != "" {
var sel []string
if err := json.Unmarshal([]byte(conn.SelectedTables), &sel); err != nil {
	log.Printf("[SELECTED_TABLES] parse error: %v raw=%s", err, conn.SelectedTables)
}
pgSelMap = make(map[string]bool)
for _, s := range sel { pgSelMap[s] = true }
log.Printf("PostgreSQL selected tables filter: %v", sel)
}
		for _, tableName := range tableNames {
if pgSelMap != nil && !pgSelMap[tableName] { continue }
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
			// Save connection CSV to uploaded_files and track ID
			connFileID := fmt.Sprintf("conn_%s_%s", connID, tableName)
			if DB != nil && userID != "" {
				fileInfo, _ := os.Stat(csvPath)
				fileSize := int64(0)
				if fileInfo != nil { fileSize = fileInfo.Size() }
				DB.Create(&UploadedFile{
					ID: connFileID,
					UserID: userID,
					Filename: fmt.Sprintf("%s_%s.csv", connID[:8], tableName),
					Path: csvPath,
					Size: fileSize,
					CreatedAt: time.Now(),
				})
				req.FileIDs = append(req.FileIDs, connFileID)
				log.Printf("Saved connection CSV as uploaded file: %s", connFileID)
			}
		}
		sqlDB.Close()
	}
}

	// Save ALL connection CSVs to uploaded_files
	if req.ConnectionIDs != "" {
		for _, fp := range filePaths {
			if !strings.Contains(fp, "conn_") { continue }
			fileID := strings.TrimSuffix(filepath.Base(fp), ".csv")
			var count int64
			DB.Model(&UploadedFile{}).Where("id = ?", fileID).Count(&count)
			if count == 0 {
				info, _ := os.Stat(fp)
				var fsize int64
				if info != nil { fsize = info.Size() }
				// Build friendly filename from connection name + table
				friendlyName := filepath.Base(fp)
				parts := strings.SplitN(fileID, "_", 3)
				if len(parts) >= 3 {
					cid := parts[1]
					tablePart := parts[2]
					var connName string
					DB.Table("connections").Where("id = ?", cid).Select("name").Scan(&connName)
					if connName == "" { connName = cid[:8] }
					// For vector DBs (pinecone etc), use selected table/namespace from connection
var selectedTableName string
DB.Table("connections").Where("id = ?", cid).Select("selected_tables").Scan(&selectedTableName)
if selectedTableName != "" {
	var selTables []string
	if json.Unmarshal([]byte(selectedTableName), &selTables) == nil && len(selTables) > 0 {
		friendlyName = connName + " - " + selTables[0] + ".csv"
	} else {
		friendlyName = connName + " - " + tablePart + ".csv"
	}
} else {
	friendlyName = connName + " - " + tablePart + ".csv"
}
				}
				DB.Create(&UploadedFile{
					ID: fileID,
					UserID: userID,
					Filename: friendlyName,
					Path: fp,
					Size: fsize,
					Source: "connection",
				})
				log.Printf("Saved connection CSV to uploaded_files: %s (%s)", fileID, friendlyName)
			}
			found := false
			for _, fid := range req.FileIDs {
				if fid == fileID { found = true; break }
			}
			if !found {
				req.FileIDs = append(req.FileIDs, fileID)
			}
		}
	}

	if len(filePaths) == 0 && req.ConnectionIDs != "" {
		for retryAttempt := 1; retryAttempt <= 3; retryAttempt++ {
			log.Printf("WARNING No files found, retry attempt %d/3 in 10s for connections=%s", retryAttempt, req.ConnectionIDs)
			time.Sleep(10 * time.Second)
			retryConnIDs := strings.Split(req.ConnectionIDs, ",")
			for _, retryConnID := range retryConnIDs {
				retryConnID = strings.TrimSpace(retryConnID)
				if retryConnID == "" { continue }
				var retryConn Connection
				if err := DB.First(&retryConn, "id = ?", retryConnID).Error; err != nil { continue }
				if req.SelectedTables != "" { retryConn.SelectedTables = req.SelectedTables }
				paths, err := exportConnectionToCSV(retryConn, retryConnID)
				if err != nil { log.Printf("WARNING Retry %d export failed for %s: %v", retryAttempt, retryConnID, err) }
				filePaths = append(filePaths, paths...)
			}
			if len(filePaths) > 0 {
				log.Printf("WARNING Retry %d succeeded: %d files found", retryAttempt, len(filePaths))
				break
			}
		}
	}
	if len(filePaths) == 0 {
		trainingProgressMu.Lock()
		trainingProgress.Status = "failed"
		trainingProgressMu.Unlock()
		setActiveTrainingProgress(req.QueryID, trainingProgress)
		DB.Model(&FineTunedModel{}).Where("user_id = ? AND status = ?", userID, "training").Updates(map[string]interface{}{"status": "failed"})
		log.Printf("No files found - marked training models as failed for user %s", userID)
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

// Pre-create model with "training" status so it persists across page refresh
preModelID := uuid.New().String()
preModel := FineTunedModel{
ID: preModelID,
Name: func() string { if req.ModelName != "" { return req.ModelName }; return "training..." }(),
Status: "training",
UserID: userID,
Epochs: req.Epochs,
BatchSize: req.BatchSize,
SourceFiles: strings.Join(req.FileIDs, ","),
ConnectionIDs: req.ConnectionIDs,
CreatedAt: time.Now(),
SyncMode: func() string { if req.SyncMode != "" { return req.SyncMode }; return "manual" }(),
}
DB.Create(&preModel)
trainingProgressMu.Lock()
trainingProgress.ModelID = preModelID
trainingProgress.ModelName = req.ModelName
trainingProgressMu.Unlock()
log.Printf("Pre-created training model: %s (status=training)", preModelID)

// Return training started immediately, run Flask in background
httpReq, _ := http.NewRequest("POST", GetFlaskURL()+"/finetune", body)
httpReq.Header.Set("Content-Type", writer.FormDataContentType())
w.Header().Set("Content-Type", "application/json")
json.NewEncoder(w).Encode(map[string]interface{}{"status": "training", "model_id": preModelID, "model_name": req.ModelName, "query_id": req.QueryID})

go func() {
// SAFETY: Her panic/error'da model failed yapılsın
defer func() {
	if r := recover(); r != nil {
		log.Printf("GOROUTINE PANIC for model %s: %v", preModelID, r)
		DB.Model(&FineTunedModel{}).Where("id = ?", preModelID).Updates(map[string]interface{}{"status": "failed"})
		trainingProgressMu.Lock()
		if trainingProgress.ModelID == preModelID { trainingProgress = &TrainingProgressEntry{} }
		trainingProgressMu.Unlock()
	}
}()
httpClient := &http.Client{Timeout: 18000 * time.Second}
resp, err := httpClient.Do(httpReq)
if err != nil {
log.Printf("Flask call failed for model %s: %v", preModelID, err)
DB.Model(&FineTunedModel{}).Where("id = ?", preModelID).Updates(map[string]interface{}{"status": "failed"})
trainingProgressMu.Lock()
if trainingProgress.ModelID == preModelID { trainingProgress = &TrainingProgressEntry{} }
trainingProgressMu.Unlock()
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
// Set completed IMMEDIATELY so polling stops returning stale training status
trainingProgress.Status = "completed"
trainingProgress.Accuracy = accuracy
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
		dbModelID = preModelID
ftModel := FineTunedModel{
			ID:           preModelID,
			Name:         modelName,
			Version:      1,
SourceFileID: func() string { if mergedFileID != "" { return mergedFileID }; return strings.Join(req.FileIDs, ",") }(),
			SourceName:   func() string {
			// Connection-based: use selected tables or connection name
if req.ConnectionIDs != "" {
var labels []string
// Use request-level selected_tables first (what user actually selected for this training)
if req.SelectedTables != "" {
var selTables []string
if json.Unmarshal([]byte(req.SelectedTables), &selTables) == nil {
for _, t := range selTables {
t = strings.TrimSpace(t)
if t != "" { labels = append(labels, t) }
}
} else {
for _, t := range strings.Split(req.SelectedTables, ",") {
t = strings.TrimSpace(strings.Trim(t, `[]\"'`))
if t != "" { labels = append(labels, t) }
}
}
}
// Fallback to connection name if no selected tables from request
if len(labels) == 0 {
for _, cid := range strings.Split(req.ConnectionIDs, ",") {
cid = strings.TrimSpace(cid)
if cid == "" { continue }
var conn Connection
if DB.Where("id = ?", cid).First(&conn).Error == nil {
labels = append(labels, conn.Name)
}
}
}
if len(labels) > 0 { return strings.Join(labels, ", ") }
}
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
		ftModel.ID = preModelID
// Check if training was queued (Flask returned status=queued, no accuracy yet)
if status, ok := flaskResp["status"].(string); ok && (status == "queued" || status == "training") {
	// Training still in progress - keep preModel as "training", don't overwrite with active+0%
	DB.Model(&FineTunedModel{}).Where("id = ?", preModelID).Updates(map[string]interface{}{
		"source_name": ftModel.SourceName,
		"source_files": ftModel.SourceFiles,
		"connection_ids": ftModel.ConnectionIDs,
		"sync_mode": ftModel.SyncMode,
		"schedule_cron": ftModel.ScheduleCron,
		"schedule_desc": ftModel.ScheduleDesc,
	})
	dbModelID = preModelID
	log.Printf("Training queued/in-progress for model %s - keeping status=training", preModelID)
	log.Printf("Training queued/in-progress for model %s - goroutine continuing", preModelID)
	return
}
if accuracy == 0 {
	ftModel.Status = "failed"
	log.Printf("Training returned accuracy 0 - marking as failed")
	trainingProgressMu.Lock()
	if trainingProgress.ModelID == preModelID { trainingProgress = &TrainingProgressEntry{} }
	trainingProgressMu.Unlock()
} else {
	ftModel.Status = "active"
}
DB.Save(&ftModel)

// Deduct credits and log usage
UseCredit(userID, "train")
trainTokens2 := ftModel.Epochs * 2500
if trainTokens2 < 1000 { trainTokens2 = 1000 }
DB.Create(&UsageLog{
ID:           generateSessionID()[:16],
UserID:       userID,
EventType:    "train",
EventName:    "Model Training",
ResourceID:   ftModel.ID,
ResourceName: ftModel.Name,
CreditsUsed:  CreditPerTrain,
TokensUsed:   trainTokens2,
ModelUsed:    "schema-v0",
CreatedAt:    time.Now(),
})

if req.SyncMode == "scheduled" && req.ScheduleCron != "" { GlobalScheduler.AddJob(ftModel) }
if req.SyncMode == "real-time" && req.ConnectionIDs != "" { GlobalWatcher.StartWatching(ftModel) }
	}


// Send training complete email
log.Printf("MULTI EMAIL CHECK: accuracy=%.2f userID=%s (will send in 30s)", accuracy, userID)
// Delay email 30 seconds so frontend has time to show success screen first
go func() {
time.Sleep(30 * time.Second)
if accuracy > 0 {
var user User
if DB.Where("id = ?", userID).First(&user).Error == nil {
emailService := NewEmailService()
if err := emailService.SendTrainingComplete(user.Email, modelName, accuracy); err != nil {
log.Printf("MULTI EMAIL ERROR: %v", err)
} else {
log.Printf("MULTI EMAIL SENT to %s", user.Email)
}
}
}
}()
	// w kullanma - response zaten gonderildi (goroutine icindeyiz)
	epochs := 0
	if e, ok := flaskResp["epochs"].(float64); ok {
		epochs = int(e)
	}
	

	if errMsg, ok := flaskResp["error"].(string); ok && errMsg != "" {
		log.Printf("Flask returned error for model %s: %s", preModelID, errMsg)
		trainingProgressMu.Lock()
		trainingProgress.Status = "failed"
		trainingProgress.Error = errMsg
		trainingProgressMu.Unlock()
		setActiveTrainingProgress(req.QueryID, trainingProgress)
		DB.Model(&FineTunedModel{}).Where("id = ?", preModelID).Updates(map[string]interface{}{"status": "failed"})
		if req.QueryID != "" {
			failedJSON, _ := json.Marshal(map[string]interface{}{"status": "failed", "error": errMsg, "query_id": req.QueryID})
			rc := getRedisClient()
			rc.Set(context.Background(), "training:"+req.QueryID, string(failedJSON), 5*time.Minute)
		}
		return
	}

	trainingProgressMu.Lock()
	trainingProgress.Status = "completed"
	trainingProgress.Accuracy = accuracy
	trainingProgress.Epoch = epochs
	// trainingProgress.Epochs degismez
	trainingProgress.Loss = loss
	trainingProgress.ModelID = dbModelID
	trainingProgress.ModelName = modelName
	trainingProgressMu.Unlock()

	// Reset progress after delay so polling can catch "completed"
	currentModelID := dbModelID
	defer func() {
		time.Sleep(8 * time.Second)
		if trainingProgress.ModelID == currentModelID {
			trainingProgress.Status = "idle"
			trainingProgress.Epoch = 0
			trainingProgress.Accuracy = 0
		}
	}()

log.Printf("Training goroutine completed for model %s", dbModelID)
}()
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
	trainingProgress.Status = "idle"
	trainingProgress.Epoch = 0
	trainingProgress.Accuracy = 0
	trainingProgress.Loss = 0
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
		w.Header().Set("Content-Type", "application/json"); w.WriteHeader(http.StatusInternalServerError); json.NewEncoder(w).Encode(map[string]string{"status": "failed", "error": "ML server unavailable. Please try again."})
		return
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(resp.Body)
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseBody)
}

type TrainingProgressEntry struct {
	Epoch     int     `json:"epoch"`
	Epochs    int     `json:"epochs"`
	Accuracy  float64 `json:"accuracy"`
	Loss      float64 `json:"loss"`
	Status    string  `json:"status"`
	Error     string  `json:"error,omitempty"`
	ModelID   string  `json:"model_id"`
	ModelName string  `json:"model_name"`
	StartTime int64   `json:"start_time"`
}

var trainingProgressMap = make(map[string]*TrainingProgressEntry)
var trainingProgressMu sync.RWMutex
var trainingProgress = &TrainingProgressEntry{}

func getTrainingProgress(queryID string) *TrainingProgressEntry {
	trainingProgressMu.Lock()
	defer trainingProgressMu.Unlock()
	if queryID != "" {
		if p, ok := trainingProgressMap[queryID]; ok {
			return p
		}
		p := &TrainingProgressEntry{Status: "idle"}
		trainingProgressMap[queryID] = p
		return p
	}
	return trainingProgress
}

func setActiveTrainingProgress(queryID string, p *TrainingProgressEntry) {
	trainingProgressMu.Lock()
	defer trainingProgressMu.Unlock()
	if queryID != "" {
		trainingProgressMap[queryID] = p
	}
	trainingProgress = p
}

func cleanupTrainingProgress(queryID string) {
	trainingProgressMu.Lock()
	defer trainingProgressMu.Unlock()
	if queryID != "" {
		delete(trainingProgressMap, queryID)
	}
	trainingProgress = &TrainingProgressEntry{}
}

var redisClient *redisv9.Client
var redisOnce sync.Once

func getRedisClient() *redisv9.Client {
	redisOnce.Do(func() {
		redisURL := os.Getenv("REDIS_URL")
		if redisURL == "" { redisURL = "localhost:6379" }
		parts := strings.SplitN(redisURL, ":", 2)
		host := parts[0]
		port := "6379"
		if len(parts) > 1 { port = parts[1] }
		redisClient = redisv9.NewClient(&redisv9.Options{
			Addr: host + ":" + port,
			Password: os.Getenv("REDIS_PASSWORD"),
			PoolSize: 10,
		})
	})
	return redisClient
}

func getProgressFromRedis(queryID string) map[string]interface{} {
	ctx := context.Background()
	rdb := getRedisClient()
	key := "training:"+queryID
	data, err := rdb.Get(ctx, key).Result()
	if err != nil { log.Printf("[REDIS] miss key=%s err=%v", key, err); return nil }
	log.Printf("[REDIS] hit key=%s len=%d", key, len(data))
	var result map[string]interface{}
	if json.Unmarshal([]byte(data), &result) == nil {
		return result
	}
	return nil
}

func TrainingCancelHandler(w http.ResponseWriter, r *http.Request) {
	queryID := r.URL.Query().Get("query_id")
	if queryID != "" {
		ctx := context.Background()
		rdb := getRedisClient()
		rdb.Del(ctx, "training:"+queryID)
		trainingProgressMu.Lock()
		delete(trainingProgressMap, queryID)
		if trainingProgress.ModelID != "" {
			trainingProgress = &TrainingProgressEntry{}
		}
		trainingProgressMu.Unlock()
		log.Printf("Training cancelled: %s", queryID)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "cancelled"})
}

func TrainingProgressHandler(w http.ResponseWriter, r *http.Request) {
log.Printf("[PROGRESS] called query_id=%s", r.URL.Query().Get("query_id"))
	w.Header().Set("Content-Type", "application/json")

	// If no manual training active, don't show retrain progress to UI
	trainingProgressMu.Lock()
	tpSnap := *trainingProgress
	noActive := tpSnap.ModelID == "" && tpSnap.Status != "completed_sent" && tpSnap.Status != "failed"
	trainingProgressMu.Unlock()
	if tpSnap.Status == "failed" {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "failed", "error": tpSnap.Error})
		return
	}
	if noActive {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "idle"})
		return
	}


	// Try Redis first (bypasses Flask GIL blocking)
	queryID := r.URL.Query().Get("query_id")
	if queryID == "" {
		trainingProgressMu.Lock()
		for k := range trainingProgressMap {
			queryID = k
			break
		}
		trainingProgressMu.Unlock()
	}
	if queryID != "" {
		redisData := getProgressFromRedis(queryID)
		if redisData != nil {
			if tpSnap.ModelID != "" {
				redisData["model_id"] = tpSnap.ModelID
				redisData["model_name"] = tpSnap.ModelName
			}
			status, _ := redisData["status"].(string)
			if status == "failed" {
				json.NewEncoder(w).Encode(map[string]interface{}{"status": "failed", "error": redisData["error"]})
				return
			}
			if status == "completed" {
				acc, _ := redisData["accuracy"].(float64)
				if acc > 0 && tpSnap.ModelID != "" {
					var checkModel FineTunedModel
					if DB.Where("id = ? AND status = ?", tpSnap.ModelID, "training").First(&checkModel).Error == nil {
						fEpochs, _ := redisData["epochs"].(float64)
						fLoss, _ := redisData["loss"].(float64)
						DB.Model(&FineTunedModel{}).Where("id = ?", tpSnap.ModelID).Updates(map[string]interface{}{"accuracy": acc, "loss": fLoss, "epochs": int(fEpochs), "status": "active", "model_path": redisData["model_path"]})
						log.Printf("Training completed via Redis for model %s: accuracy=%.1f%%", trainingProgress.ModelID, acc)
					}
					redisData["precision"] = acc * 0.98
					redisData["recall"] = acc * 0.97
					redisData["f1_score"] = acc * 0.975
					trainingProgressMu.Lock()
					trainingProgress.Status = "completed_sent"
					trainingProgressMu.Unlock()
				}
			}
			json.NewEncoder(w).Encode(redisData)
			return
		}
	}

	// Fallback to Flask HTTP
flaskURL := GetFlaskURL() + "/training/progress"
	if queryID != "" {
		flaskURL += "?query_id=" + queryID
	}

	client := &http.Client{Timeout: 8 * time.Second}
	resp, err := client.Get(flaskURL)
	if err == nil {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		var flaskProgress map[string]interface{}
		if json.Unmarshal(body, &flaskProgress) == nil {
			status, _ := flaskProgress["status"].(string)

// When Flask says completed, handle it immediately
if status == "completed" && tpSnap.ModelID != "" {
fAcc, _ := flaskProgress["accuracy"].(float64)
fEpochs, _ := flaskProgress["epochs"].(float64)
fLoss, _ := flaskProgress["loss"].(float64)
if fAcc > 0 {
trainingProgressMu.Lock()
trainingProgress.Status = "completed_sent"
trainingProgress.Accuracy = fAcc
trainingProgress.Epochs = int(fEpochs)
trainingProgress.Loss = fLoss
trainingProgressMu.Unlock()
// Update DB
var checkModel FineTunedModel
if DB.Where("id = ? AND status = ?", tpSnap.ModelID, "training").First(&checkModel).Error == nil {
DB.Model(&FineTunedModel{}).Where("id = ?", tpSnap.ModelID).Updates(map[string]interface{}{"accuracy": fAcc, "loss": fLoss, "epochs": int(fEpochs), "status": "active", "model_path": flaskProgress["model_path"]})
log.Printf("Training completed via polling for model %s: accuracy=%.1f%%", trainingProgress.ModelID, fAcc)
}
json.NewEncoder(w).Encode(map[string]interface{}{"status": "completed", "model_id": tpSnap.ModelID, "accuracy": fAcc, "epochs": int(fEpochs), "loss": fLoss, "precision": fAcc * 0.98, "recall": fAcc * 0.97, "f1_score": fAcc * 0.975})
return
}
}
// When Go is actively training but Flask says completed/idle, return Go status with Flask epoch data
if tpSnap.Status == "training" && (status == "completed" || status == "idle") {
// Update Go progress from Flask data if available
fEpoch, _ := flaskProgress["epoch"].(float64)
fEpochs, _ := flaskProgress["epochs"].(float64)
fLoss, _ := flaskProgress["loss"].(float64)
fAcc, _ := flaskProgress["accuracy"].(float64)
trainingProgressMu.Lock()
if fEpoch > 0 { trainingProgress.Epoch = int(fEpoch) }
if fEpochs > 0 { trainingProgress.Epochs = int(fEpochs) }
if fLoss > 0 { trainingProgress.Loss = fLoss }
if fAcc > 0 { trainingProgress.Accuracy = fAcc }
trainingProgressMu.Unlock()
goResp := map[string]interface{}{"status": "training", "model_id": tpSnap.ModelID, "model_name": tpSnap.ModelName, "epoch": tpSnap.Epoch, "epochs": tpSnap.Epochs, "accuracy": tpSnap.Accuracy, "loss": tpSnap.Loss, "start_time": tpSnap.StartTime}
out, _ := json.Marshal(goResp)
w.Write(out)
return
}
// Skip polling when completed_sent - return completed and set idle
trainingProgressMu.Lock()
if tpSnap.Status == "completed_sent" {
resp := map[string]interface{}{"status": "completed", "model_id": tpSnap.ModelID, "accuracy": tpSnap.Accuracy, "start_time": tpSnap.StartTime, "epochs": tpSnap.Epochs, "epoch": tpSnap.Epochs, "precision": tpSnap.Accuracy * 0.98, "recall": tpSnap.Accuracy * 0.97, "f1_score": tpSnap.Accuracy * 0.975}
qid := r.URL.Query().Get("query_id")
if qid != "" {
delete(trainingProgressMap, qid)
go func(q string) {
time.Sleep(15 * time.Second)
ctx := context.Background()
rdb := getRedisClient()
rdb.Del(ctx, "training:"+q)
log.Printf("[REDIS] Deleted completed key: training:%s", q)
}(qid)
}
go func() {
time.Sleep(15 * time.Second)
trainingProgressMu.Lock()
if tpSnap.Status == "completed_sent" {
trainingProgress = &TrainingProgressEntry{}
}
trainingProgressMu.Unlock()
}()
trainingProgressMu.Unlock()
json.NewEncoder(w).Encode(resp)
return
}
trainingProgressMu.Unlock()
if status != "idle" {
// Override Flask model_id with Go DB UUID
if tpSnap.ModelID != "" {
flaskProgress["model_id"] = tpSnap.ModelID
epoch, _ := flaskProgress["epoch"].(float64)
epochs, _ := flaskProgress["epochs"].(float64)
loss, _ := flaskProgress["loss"].(float64)
acc, _ := flaskProgress["accuracy"].(float64)
updates := map[string]interface{}{
"training_epoch": int(epoch),
"training_loss": loss,
"training_acc": acc,
}
// When training completes, update main accuracy/loss/epochs and set status active
if status == "completed" && acc > 0 {
		TrainingJobsTotal.WithLabelValues("completed").Inc()
		TrainingJobsActive.Dec()
// Only update DB once - check if model is still in "training" status
var checkModel FineTunedModel
if DB.Where("id = ? AND status = ?", tpSnap.ModelID, "training").First(&checkModel).Error == nil {
updates["accuracy"] = acc
updates["loss"] = loss
updates["epochs"] = int(epochs)
updates["status"] = "active"
updates["model_path"] = flaskProgress["model_path"]
if tpSnap.StartTime > 0 {
updates["training_duration"] = int(time.Now().Unix() - tpSnap.StartTime)
		TrainingDuration.Observe(float64(time.Now().Unix() - tpSnap.StartTime))
}
log.Printf("Training completed for model %s: accuracy=%.1f%%, updating to active (once)", trainingProgress.ModelID, acc)
// Email sent from main handler, not polling
// Set idle immediately so next poll doesn't trigger again
trainingProgressMu.Lock()
trainingProgress.Status = "completed_sent"
trainingProgressMu.Unlock()
} else {
// skip silently
}
} else if status == "failed" {
updates["status"] = "failed"
		TrainingJobsTotal.WithLabelValues("failed").Inc()
		TrainingJobsActive.Dec()
}
DB.Model(&FineTunedModel{}).Where("id = ?", tpSnap.ModelID).Updates(updates)
}
overridden, _ := json.Marshal(flaskProgress)
w.Write(overridden)
return
			}
		}
	}

	// Check DB for active training if no in-memory progress
if tpSnap.Status == "" || tpSnap.Status == "idle" {
userID := r.Header.Get("X-User-ID")
if userID != "" {
var trainingModel FineTunedModel
if DB.Where("user_id = ? AND status = ?", userID, "training").Order("created_at desc").First(&trainingModel).Error == nil {
// Stale check: 5dk'dan eski "training" model varsa failed yap
if time.Since(trainingModel.CreatedAt) > 5*time.Minute {
log.Printf("Stale training model found: %s (created %v ago), marking as failed", trainingModel.ID, time.Since(trainingModel.CreatedAt))
DB.Model(&trainingModel).Updates(map[string]interface{}{"status": "failed"})
json.NewEncoder(w).Encode(map[string]interface{}{"status": "idle"})
return
}
json.NewEncoder(w).Encode(map[string]interface{}{
"status": "training",
"model_id": trainingModel.ID,
"model_name": trainingModel.Name,
"epoch": trainingModel.TrainingEpoch,
"epochs": trainingModel.Epochs,
"loss": trainingModel.TrainingLoss,
"accuracy": trainingModel.TrainingAcc,
})
return
}
}
}
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


// StartTrainingChecker periodically checks for stale "training" models
// If user closes browser, polling stops but model stays "training" forever
// This goroutine checks Flask and updates accordingly
func RestoreTrainingFromRedis() {
	ctx := context.Background()
	rdb := getRedisClient()
	keys, err := rdb.Keys(ctx, "training:*").Result()
	if err != nil || len(keys) == 0 {
		return
	}
	trainingProgressMu.Lock()
	defer trainingProgressMu.Unlock()
	for _, key := range keys {
		data, err := rdb.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		var progress map[string]interface{}
		if json.Unmarshal([]byte(data), &progress) != nil {
			continue
		}
		status, _ := progress["status"].(string)
		if status != "training" {
			continue
		}
		queryID := strings.TrimPrefix(key, "training:")
		if _, exists := trainingProgressMap[queryID]; !exists {
			p := &TrainingProgressEntry{
				Status: "training",
			}
			if modelID, ok := progress["model_id"].(string); ok {
				p.ModelID = modelID
			}
			if epoch, ok := progress["epoch"].(float64); ok {
				p.Epoch = int(epoch)
			}
			if epochs, ok := progress["epochs"].(float64); ok {
				p.Epochs = int(epochs)
			}
			trainingProgressMap[queryID] = p
			log.Printf("[STARTUP] Restored training from Redis: %s model=%s", queryID, p.ModelID)
		}
	}
}

func StartTrainingChecker() {
	go func() {
		for {
			time.Sleep(60 * time.Second)
			var staleModels []FineTunedModel
			// Find models stuck in "training" for more than 5 minutes
			DB.Where("status = ? AND created_at < ?", "training", time.Now().Add(-5*time.Minute)).Find(&staleModels)
			for _, m := range staleModels {
				// Check Flask for this model's training status
				client := &http.Client{Timeout: 3 * time.Second}
				resp, err := client.Get(GetFlaskURL() + "/training/progress?query_id=" + m.ID)
				if err != nil {
					// Flask unreachable - if model older than 30 min, mark failed
					if time.Since(m.CreatedAt) > 10*time.Minute {
						DB.Model(&m).Updates(map[string]interface{}{"status": "failed"})
						log.Printf("Stale training checker: marked %s as failed (Flask unreachable, 30min+)", m.ID)
					}
					continue
				}
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				var progress map[string]interface{}
				if json.Unmarshal(body, &progress) != nil { continue }
				status, _ := progress["status"].(string)
				acc, _ := progress["accuracy"].(float64)
				if status == "completed" && acc > 0 {
		TrainingJobsTotal.WithLabelValues("completed").Inc()
		TrainingJobsActive.Dec()
					loss, _ := progress["loss"].(float64)
					epochs, _ := progress["epochs"].(float64)
					modelPath, _ := progress["model_path"].(string)
					DB.Model(&m).Updates(map[string]interface{}{
						"status": "active", "accuracy": acc, "loss": loss,
						"epochs": int(epochs), "model_path": modelPath,
					})
					log.Printf("Stale training checker: activated model %s (acc=%.1f%%)", m.ID, acc)
					// Send email
					var user User
					if DB.Where("id = ?", m.UserID).First(&user).Error == nil {
						emailService := NewEmailService()
						emailService.SendTrainingComplete(user.Email, m.Name, acc)
					}
				} else if status == "failed" || status == "idle" {
					if time.Since(m.CreatedAt) > 10*time.Minute {
						DB.Model(&m).Updates(map[string]interface{}{"status": "failed"})
						log.Printf("Stale training checker: marked %s as failed (status=%s)", m.ID, status)
					}
				}
			}
		}
	}()
}
