package handlers

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"path/filepath"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"schemalabsai/services"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	awssession "github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
sf "github.com/snowflakedb/gosnowflake"
	gormpg "gorm.io/driver/postgres"
	gormmysql "gorm.io/driver/mysql"
	"gorm.io/gorm"
)

// ============================================================
// SCHEDULER - Manages scheduled model retraining
// ============================================================

type Scheduler struct {
	mu       sync.Mutex
	jobs     map[string]*ScheduledJob
	stopChan chan struct{}
	running  bool
}

type ScheduledJob struct {
	ModelID   string
	UserID    string
	CronExpr  string
	NextRun   time.Time
	LastRun   *time.Time
	Status    string
	FileIDs   []string
	ModelName string
	Epochs    int
}

var GlobalScheduler = &Scheduler{
	jobs:     make(map[string]*ScheduledJob),
	stopChan: make(chan struct{}),
}

func (s *Scheduler) Start() {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.mu.Unlock()

	log.Println("🕐 Scheduler started")
	s.LoadFromDB()

	// Load real-time watchers too
	var rtModels []FineTunedModel
	DB.Where("sync_mode = ?", "real-time").Find(&rtModels)
	for _, m := range rtModels {
		GlobalWatcher.StartWatching(m)
	}
	if len(rtModels) > 0 {
		log.Printf("👁️ Loaded %d real-time watchers", len(rtModels))
	}

	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				s.tick()
			case <-s.stopChan:
				return
			}
		}
	}()
}

func (s *Scheduler) Stop() { close(s.stopChan) }

func (s *Scheduler) LoadFromDB() {
	var models []FineTunedModel
	DB.Where("sync_mode = ? AND schedule_cron != ''", "scheduled").Find(&models)

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, m := range models {
		nextRun := calculateNextRun(m.ScheduleCron)
		s.jobs[m.ID] = &ScheduledJob{
			ModelID: m.ID, UserID: m.UserID, CronExpr: m.ScheduleCron,
			NextRun: nextRun, Status: "idle",
			FileIDs: strings.Split(m.SourceFiles, ","),
			ModelName: m.Name, Epochs: m.Epochs,
		}
	}
	log.Printf("📅 Loaded %d scheduled jobs", len(s.jobs))
}

func (s *Scheduler) AddJob(model FineTunedModel) {
	s.mu.Lock()
	defer s.mu.Unlock()
	nextRun := calculateNextRun(model.ScheduleCron)
	s.jobs[model.ID] = &ScheduledJob{
		ModelID: model.ID, UserID: model.UserID, CronExpr: model.ScheduleCron,
		NextRun: nextRun, Status: "idle",
		FileIDs: strings.Split(model.SourceFiles, ","),
		ModelName: model.Name, Epochs: model.Epochs,
	}
	log.Printf("📅 Job added: %s next: %s", model.Name, nextRun.Format("2006-01-02 15:04"))
}

func (s *Scheduler) RemoveJob(modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.jobs, modelID)
}

func (s *Scheduler) tick() {
	s.mu.Lock()
	var due []*ScheduledJob
	now := time.Now()
	for _, job := range s.jobs {
		if job.Status != "running" && now.After(job.NextRun) {
			due = append(due, job)
		}
	}
	s.mu.Unlock()
	for _, job := range due {
		go s.ExecuteJob(job)
	}
}

func (s *Scheduler) ExecuteJob(job *ScheduledJob) {
	s.mu.Lock()
	if job.Status == "running" {
		s.mu.Unlock()
		log.Printf("[SCHEDULER] Job already running, skipping: %s", job.ModelName)
		return
	}
	s.mu.Unlock()
	if ok, reason := CheckCredits(job.UserID, CreditPerTrain); !ok {
		log.Printf("Retrain blocked for %s: %s", job.ModelName, reason)
		DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Update("sync_status", "error")
		return
	}
	if ok, reason := CheckStorage(job.UserID, 0); !ok {
		log.Printf("Retrain blocked for %s: %s", job.ModelName, reason)
		DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Update("sync_status", "error")
		return
	}
	s.mu.Lock()
	job.Status = "running"
	s.mu.Unlock()

	log.Printf("\xf0\x9f\x94\x84 Retrain starting: %s", job.ModelName)
	now := time.Now()
	DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Updates(map[string]interface{}{
		"sync_status": "syncing", "last_sync_at": now,
	})

	// Step 1: Refresh data from connections (fetch latest data)
	var model FineTunedModel
	if DB.Where("id = ?", job.ModelID).First(&model).Error == nil && model.ConnectionIDs != "" {
		newFileIDs := refreshDataFromConnections(model)
		if len(newFileIDs) > 0 {
			job.FileIDs = newFileIDs
			DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Update("source_files", strings.Join(newFileIDs, ","))
			log.Printf("📥 Refreshed %d data files from connections", len(newFileIDs))
			// Update storage usage
			var totalSize int64
			DB.Model(&UploadedFile{}).Where("user_id = ?", job.UserID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
			storageMB := float64(totalSize) / (1024 * 1024)
			var schConns []Connection
			DB.Where("user_id = ?", job.UserID).Find(&schConns)
			for _, c := range schConns {
				if c.CachedTables != "" && c.CachedTables != "null" && c.CachedTables != "[]" {
					var cached struct{ TableDetails []struct{ Rows int `json:"rows"`; Columns int `json:"columns"` } `json:"table_details"` }
					if json.Unmarshal([]byte(c.CachedTables), &cached) == nil {
						for _, t := range cached.TableDetails {
							cols := t.Columns; if cols < 10 { cols = 10 }
							storageMB += float64(t.Rows * cols * 20) / (1024 * 1024)
						}
					}
				}
			}
			DB.Model(&UserQuota{}).Where("user_id = ?", job.UserID).Update("storage_used_mb", storageMB)
		}
	}
	// Step 2: Retrain with fresh data
	err := triggerRetrain(job)

	s.mu.Lock()
	if err != nil {
		job.Status = "error"
		log.Printf("❌ Retrain failed: %s - %v", job.ModelName, err)
		DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Update("sync_status", "error")
	} else {
		job.Status = "idle"
		job.LastRun = &now
		job.NextRun = calculateNextRun(job.CronExpr)
		log.Printf("✅ Retrain complete: %s, next: %s", job.ModelName, job.NextRun.Format("2006-01-02 15:04"))
		DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Updates(map[string]interface{}{
			"sync_status": "idle", "next_sync_at": job.NextRun,
		})
	}
	s.mu.Unlock()
}

func triggerRetrain(job *ScheduledJob) error {
	// Try Airflow incremental_learning DAG first
	airflowURL := os.Getenv("AIRFLOW_URL")
	if airflowURL == "" { airflowURL = "http://airflow:8080" }
	airflowUser := os.Getenv("AIRFLOW_ADMIN_USER"); if airflowUser == "" { airflowUser = "admin" }
	airflowPass := os.Getenv("AIRFLOW_ADMIN_PASSWORD")

	fileIDsJSON := "["
	for i, fid := range job.FileIDs {
		if i > 0 { fileIDsJSON += "," }
		fileIDsJSON += `"` + strings.TrimSpace(fid) + `"`
	}
	fileIDsJSON += "]"

	payload := fmt.Sprintf(`{"conf":{"model_id":"%s","user_id":"%s","file_ids":%s,"existing_file_ids":%s,"new_file_ids":[]}}`,
		job.ModelID, job.UserID, fileIDsJSON, fileIDsJSON)

	airflowReq, _ := http.NewRequest("POST", airflowURL+"/api/v1/dags/incremental_learning/dagRuns", strings.NewReader(payload))
	airflowReq.Header.Set("Content-Type", "application/json")
	airflowReq.SetBasicAuth(airflowUser, airflowPass)
	httpCl := &http.Client{Timeout: 10 * time.Second}
	airflowResp, aerr := httpCl.Do(airflowReq)
	if aerr == nil && airflowResp.StatusCode < 300 {
		airflowResp.Body.Close()
		log.Printf("[SCHEDULER] Airflow incremental_learning triggered for model %s", job.ModelID)
		return nil
	}
	if airflowResp != nil { airflowResp.Body.Close() }
	log.Printf("[SCHEDULER] Airflow failed (%v), falling back to direct Flask", aerr)

	flaskURL := GetFlaskURL()


	// Collect file paths
	var filePaths []string
	for _, fileID := range job.FileIDs {
		fileID = strings.TrimSpace(fileID)
		if fileID == "" { continue }
		matches, _ := filepath.Glob("./uploads/" + fileID + "_*")
// Try storage if local not found
if len(matches) == 0 {
var dbFile UploadedFile
if DB != nil && DB.Where("id = ?", fileID).First(&dbFile).Error == nil && dbFile.Path != "" {
if sr, serr := services.DefaultStorage.Download(dbFile.Path); serr == nil {
tmpPath := "./uploads/tmp_sched_" + fileID
tf, _ := os.Create(tmpPath)
io.Copy(tf, sr)
tf.Close()
sr.Close()
matches = []string{tmpPath}
}
}
}
		if len(matches) > 0 {
			filePaths = append(filePaths, matches[0])
		} else {
			// Try exact path
			matches2, _ := filepath.Glob("./uploads/" + fileID + "*")
			if len(matches2) == 0 {
				var dbFile2 UploadedFile
				if DB != nil && DB.Where("id = ?", fileID).First(&dbFile2).Error == nil && dbFile2.Path != "" {
					if sr2, serr2 := services.DefaultStorage.Download(dbFile2.Path); serr2 == nil {
						tmpPath2 := "./uploads/tmp_sched2_" + fileID
						tf2, _ := os.Create(tmpPath2)
						io.Copy(tf2, sr2)
						tf2.Close()
						sr2.Close()
						matches2 = []string{tmpPath2}
					}
				}
			}
			if len(matches2) > 0 {
				filePaths = append(filePaths, matches2[0])
			}
		}
	}

	if len(filePaths) == 0 {
		return fmt.Errorf("no files found for retrain")
	}

	// Create multipart form (same as MultiTrainHandler)
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	for _, fp := range filePaths {
		file, err := os.Open(fp)
		if err != nil { continue }
		part, _ := writer.CreateFormFile("file", filepath.Base(fp))
		io.Copy(part, file)
		file.Close()
	}

	epochsField, _ := writer.CreateFormField("epochs")
	fmt.Fprintf(epochsField, "%d", job.Epochs)
	batchField, _ := writer.CreateFormField("batch_size")
	fmt.Fprintf(batchField, "64")
	lrField, _ := writer.CreateFormField("learning_rate")
	fmt.Fprintf(lrField, "0.001")
	warmupField, _ := writer.CreateFormField("warmup_steps")
	fmt.Fprintf(warmupField, "100")
	mergeField, _ := writer.CreateFormField("merge_files")
	mergeField.Write([]byte("true"))
qidField, _ := writer.CreateFormField("query_id")
qidField.Write([]byte("retrain-" + job.ModelID))
	writer.Close()

	client := &http.Client{Timeout: 18000 * time.Second}
	req, _ := http.NewRequest("POST", flaskURL+"/finetune", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	if status, ok := result["status"].(string); ok && status == "success" {
		if acc, ok := result["accuracy"].(float64); ok {
			updates := map[string]interface{}{"accuracy": acc}
			if mp, ok := result["model_path"].(string); ok {
				updates["model_path"] = mp
			}
			DB.Model(&FineTunedModel{}).Where("id = ?", job.ModelID).Updates(updates)
		}
		return nil
	}
	if e, ok := result["error"].(string); ok {
		return fmt.Errorf("%s", e)
	}
	return fmt.Errorf("unknown error")
}

func calculateNextRun(cronExpr string) time.Time {
if cronExpr == "" {
return time.Now().Add(24 * time.Hour)
}
	now := time.Now()
	switch cronExpr {
	case "hourly":
		return now.Add(1 * time.Hour).Truncate(time.Hour)
	case "daily":
		next := time.Date(now.Year(), now.Month(), now.Day(), 2, 0, 0, 0, now.Location())
		if next.Before(now) { next = next.Add(24 * time.Hour) }
		return next
	case "weekly":
		next := now
		for next.Weekday() != time.Monday { next = next.Add(24 * time.Hour) }
		next = time.Date(next.Year(), next.Month(), next.Day(), 2, 0, 0, 0, next.Location())
		if next.Before(now) { next = next.Add(7 * 24 * time.Hour) }
		return next
	case "monthly":
		next := time.Date(now.Year(), now.Month(), 1, 2, 0, 0, 0, now.Location())
		if next.Before(now) { next = next.AddDate(0, 1, 0) }
		return next
	default:
		// Custom intervals: "1h", "2d", "1w", "1m" or "every Xh/Xd/Xw/Xm"
		expr := cronExpr
		if strings.HasPrefix(expr, "every ") {
			expr = strings.TrimSpace(strings.TrimPrefix(expr, "every "))
		}
		var val int
		unit := expr[len(expr)-1:]
		fmt.Sscanf(expr, "%d", &val)
		if val <= 0 { val = 1 }
		switch unit {
		case "h":
			return now.Add(time.Duration(val) * time.Hour)
		case "d":
			return now.Add(time.Duration(val) * 24 * time.Hour)
		case "w":
			return now.Add(time.Duration(val) * 7 * 24 * time.Hour)
		case "m":
			return now.AddDate(0, val, 0)
		}
		next := time.Date(now.Year(), now.Month(), now.Day(), 2, 0, 0, 0, now.Location())
		if next.Before(now) { next = next.Add(24 * time.Hour) }
		return next
	}
}

// ============================================================
// REAL-TIME WATCHER - Monitors ALL connection types for changes
// ============================================================

type RealTimeWatcher struct {
	mu       sync.Mutex
	watchers map[string]*ConnWatcher
}

type ConnWatcher struct {
	ModelID      string
	ConnectionID string
	ConnType     string
	LastChecksum string
	StopChan     chan struct{}
}

var GlobalWatcher = &RealTimeWatcher{
	watchers: make(map[string]*ConnWatcher),
}

func (rw *RealTimeWatcher) StartWatching(model FineTunedModel) {
	connIDs := strings.Split(model.ConnectionIDs, ",")
	for _, connID := range connIDs {
		connID = strings.TrimSpace(connID)
		if connID == "" { continue }

		var conn Connection
		if DB.Where("id = ?", connID).First(&conn).Error != nil { continue }

		w := &ConnWatcher{
			ModelID: model.ID, ConnectionID: connID,
			ConnType: conn.SubType, StopChan: make(chan struct{}),
		}

		rw.mu.Lock()
		key := model.ID + ":" + connID
		if existing, ok := rw.watchers[key]; ok { close(existing.StopChan) }
		rw.watchers[key] = w
		rw.mu.Unlock()

		go rw.poll(w, conn)
		log.Printf("👁️ Watching: model=%s conn=%s (%s)", model.Name, conn.Name, conn.SubType)
	}
}

func (rw *RealTimeWatcher) StopWatching(modelID string) {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	for key, w := range rw.watchers {
		if strings.HasPrefix(key, modelID+":") {
			close(w.StopChan)
			delete(rw.watchers, key)
		}
	}
}

func (rw *RealTimeWatcher) poll(w *ConnWatcher, conn Connection) {
ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			checksum := getConnectionChecksum(conn)
			if checksum != "" && checksum != w.LastChecksum && w.LastChecksum != "" {
				log.Printf("🔄 Data changed: model=%s conn=%s (%s)", w.ModelID, w.ConnectionID, w.ConnType)
				w.LastChecksum = checksum

				var model FineTunedModel
				if DB.Where("id = ?", w.ModelID).First(&model).Error == nil {
					job := &ScheduledJob{
						ModelID: model.ID, UserID: model.UserID,
						FileIDs: strings.Split(model.SourceFiles, ","),
						ModelName: model.Name, Epochs: model.Epochs,
					}
					go GlobalScheduler.ExecuteJob(job)
				}
			} else if w.LastChecksum == "" {
				w.LastChecksum = checksum
			}
		case <-w.StopChan:
			return
		}
	}
}

// getConnectionChecksum - detects changes for ALL connection types
func getConnectionChecksum(conn Connection) string {
	switch conn.SubType {

	// === DATABASES ===
	case "postgresql", "supabase":
		sslMode := "disable"
		if conn.SubType == "supabase" || conn.SSL { sslMode = "require" }
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			conn.Host, conn.Port, conn.Username, conn.Password, conn.Database, sslMode)
		db, err := sql.Open("postgres", dsn)
		if err != nil { return "" }
		defer db.Close()
		var count int64
		var txid string
		if conn.SelectedTables != "" {
var sel []string
json.Unmarshal([]byte(conn.SelectedTables), &sel)
if len(sel) > 0 {
placeholders := make([]string, len(sel))
args := make([]interface{}, len(sel))
for i, s := range sel { placeholders[i] = fmt.Sprintf("$%d", i+1); args[i] = s }
db.QueryRow("SELECT COALESCE(SUM(n_live_tup), 0) FROM pg_stat_user_tables WHERE relname IN ("+strings.Join(placeholders, ",")+")", args...).Scan(&count)
}
} else {
db.QueryRow("SELECT COALESCE(SUM(n_live_tup), 0) FROM pg_stat_user_tables").Scan(&count)
}
		db.QueryRow("SELECT txid_current()::text").Scan(&txid)
		return fmt.Sprintf("pg-%d-%s", count, txid)

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		db, err := sql.Open("mysql", dsn)
		if err != nil { return "" }
		defer db.Close()
		var count int64
		db.QueryRow("SELECT COALESCE(SUM(TABLE_ROWS), 0) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?", conn.Database).Scan(&count)
		var checksum string
		db.QueryRow("SELECT CONCAT(NOW(), '-', @@global.gtid_executed)").Scan(&checksum)
		return fmt.Sprintf("mysql-%d-%s", count, checksum)

	case "mongodb":
		var mongoURI string
		if conn.Endpoint != "" {
			mongoURI = conn.Endpoint
		} else {
			mongoURI = fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
			if conn.Username == "" {
				mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database)
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI))
		if err != nil { return "" }
		defer client.Disconnect(ctx)

		// Get all collection stats
		db := client.Database(conn.Database)
		collections, err := db.ListCollectionNames(ctx, bson.M{})
		if err != nil { return "" }
		totalDocs := int64(0)
		var selMap map[string]bool
		if conn.SelectedTables != "" {
			var sel []string
			json.Unmarshal([]byte(conn.SelectedTables), &sel)
			selMap = make(map[string]bool)
			for _, s := range sel { selMap[s] = true }
		}
		for _, coll := range collections {
			if selMap != nil && !selMap[coll] { continue }
			count, _ := db.Collection(coll).CountDocuments(ctx, bson.M{})
			totalDocs += count
		}
		return fmt.Sprintf("mongo-%d-%d", len(collections), totalDocs)

	case "snowflake":
		// Snowflake via REST API
		if conn.Endpoint == "" { return "" }
		return httpChecksum(conn.Endpoint+"/api/v2/statements", conn.APIKey, "Bearer")

	case "databricks":
		if conn.Endpoint == "" { return "" }
		return httpChecksum(conn.Endpoint+"/api/2.0/sql/statements", conn.APIKey, "Bearer")

	// === VECTOR DATABASES ===
	case "pinecone":
		if conn.Endpoint == "" || conn.APIKey == "" { return "" }
		return httpChecksumWithHeader(conn.Endpoint+"/describe_index_stats", "Api-Key", conn.APIKey)

	case "weaviate":
		if conn.Endpoint == "" { return "" }
		url := conn.Endpoint + "/v1/meta"
		return httpChecksum(url, conn.APIKey, "Bearer")

	case "chroma":
		if conn.Endpoint == "" { return "" }
		return httpChecksum(conn.Endpoint+"/api/v1/collections", conn.APIKey, "Bearer")

	case "lancedb":
		if conn.Endpoint == "" { return "" }
		return httpChecksum(conn.Endpoint+"/v1/table", conn.APIKey, "Bearer")

	case "rest_api", "rest":
		if conn.Endpoint == "" { return "" }
		return httpChecksum(conn.Endpoint, conn.APIKey, "Bearer")

	case "graphql":
		if conn.Endpoint == "" { return "" }
		return httpPostChecksum(conn.Endpoint, `{"query":"{ __schema { types { name } } }"}`, conn.APIKey)

	case "google-drive", "google_drive":
		if conn.APIKey == "" { return "" }
		return httpChecksum("https://www.googleapis.com/drive/v3/files?pageSize=100&orderBy=modifiedTime+desc&fields=files(id,modifiedTime)", conn.APIKey, "Bearer")

	case "aws-s3", "aws_s3":
		if conn.APIKey == "" || conn.Bucket == "" { return "" }
		sess, err := awssession.NewSession(&aws.Config{
			Region:      aws.String(conn.Region),
			Credentials: credentials.NewStaticCredentials(conn.APIKey, conn.Password, ""),
		})
		if err != nil { return "" }
		s3Client := s3.New(sess)
		result, err := s3Client.ListObjectsV2(&s3.ListObjectsV2Input{
			Bucket:  aws.String(conn.Bucket),
			MaxKeys: aws.Int64(1000),
		})
		if err != nil { return "" }
		totalSize := int64(0)
		latestMod := ""
		for _, obj := range result.Contents {
			totalSize += *obj.Size
			mod := obj.LastModified.Format(time.RFC3339)
			if mod > latestMod { latestMod = mod }
		}
		return fmt.Sprintf("s3-%d-%d-%s", *result.KeyCount, totalSize, latestMod)

	case "gcs":
		if conn.APIKey == "" || conn.Bucket == "" { return "" }
		url := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?maxResults=100", conn.Bucket)
		return httpChecksum(url, conn.APIKey, "Bearer")



	default:
		return ""
	}
}

// === HTTP HELPERS for change detection ===

func httpChecksum(url, apiKey, authType string) string {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil { return "" }
	if apiKey != "" {
		req.Header.Set("Authorization", authType+" "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	h := sha256.Sum256(body)
	return fmt.Sprintf("%x", h[:8])
}

func httpChecksumWithHeader(url, headerKey, headerVal string) string {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil { return "" }
	req.Header.Set(headerKey, headerVal)
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	h := sha256.Sum256(body)
	return fmt.Sprintf("%x", h[:8])
}

func httpPostChecksum(url, bodyStr, apiKey string) string {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("POST", url, bytes.NewBufferString(bodyStr))
	if err != nil { return "" }
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	h := sha256.Sum256(body)
	return fmt.Sprintf("%x", h[:8])
}

// ============================================================
// DATA REFRESH - Fetch latest data from connections before retrain
// ============================================================

func refreshDataFromConnections(model FineTunedModel) []string {
	connIDs := strings.Split(model.ConnectionIDs, ",")
	var newFileIDs []string

	for _, connID := range connIDs {
		connID = strings.TrimSpace(connID)
		if connID == "" { continue }

		var conn Connection
		if DB.Where("id = ?", connID).First(&conn).Error != nil { continue }

		fileIDs := fetchConnectionData(conn, model.UserID)
		newFileIDs = append(newFileIDs, fileIDs...)
	}

	// If no connection data fetched, keep original files
	if len(newFileIDs) == 0 {
		return strings.Split(model.SourceFiles, ",")
	}
	return newFileIDs
}

// fetchConnectionData - pulls fresh data from a connection and saves as CSV
func fetchConnectionData(conn Connection, userID string) []string {
	var fileIDs []string

	switch conn.SubType {
	case "postgresql", "supabase":
		sslMode := "disable"
		if conn.SubType == "supabase" || conn.SSL { sslMode = "require" }
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			conn.Host, conn.Port, conn.Username, conn.Password, conn.Database, sslMode)
		tempDB, err := gorm.Open(gormpg.Open(dsn), &gorm.Config{})
		if err != nil { log.Printf("❌ PG connect failed: %v", err); return nil }
		sqlDB, _ := tempDB.DB()
		defer sqlDB.Close()

		// Get all tables
		rows, err := sqlDB.Query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
		if err != nil { return nil }
		var tables []string
		for rows.Next() { var t string; rows.Scan(&t); tables = append(tables, t) }
		rows.Close()

		var pgSelMap map[string]bool
		if conn.SelectedTables != "" {
			var sel []string
			json.Unmarshal([]byte(conn.SelectedTables), &sel)
			pgSelMap = make(map[string]bool)
			for _, s := range sel { pgSelMap[s] = true }
		}
		for _, table := range tables {
			if pgSelMap != nil && !pgSelMap[table] { continue }
			fid := exportTableToCSV(sqlDB, table, conn, userID)
			if fid != "" { fileIDs = append(fileIDs, fid) }
		}

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		tempDB, err := gorm.Open(gormmysql.Open(dsn), &gorm.Config{})
		if err != nil { log.Printf("❌ MySQL connect failed: %v", err); return nil }
		sqlDB, _ := tempDB.DB()
		defer sqlDB.Close()

		rows, err := sqlDB.Query("SELECT table_name FROM information_schema.tables WHERE table_schema = ?", conn.Database)
		if err != nil { return nil }
		var tables []string
		for rows.Next() { var t string; rows.Scan(&t); tables = append(tables, t) }
		rows.Close()

		var mySelMap map[string]bool
		if conn.SelectedTables != "" {
			var sel []string
			json.Unmarshal([]byte(conn.SelectedTables), &sel)
			mySelMap = make(map[string]bool)
			for _, s := range sel { mySelMap[s] = true }
		}
		for _, table := range tables {
			if mySelMap != nil && !mySelMap[table] { continue }
			fid := exportTableToCSV(sqlDB, table, conn, userID)
			if fid != "" { fileIDs = append(fileIDs, fid) }
		}

	case "mongodb":
		var mongoURI string
		if conn.Endpoint != "" {
			mongoURI = conn.Endpoint
		} else {
			mongoURI = fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
			if conn.Username == "" {
				mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database)
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI))
		if err != nil { return nil }
		defer client.Disconnect(ctx)

		db := client.Database(conn.Database)
		collections, _ := db.ListCollectionNames(ctx, bson.M{})
		var mgoSelMap map[string]bool
		if conn.SelectedTables != "" {
			var sel []string
			json.Unmarshal([]byte(conn.SelectedTables), &sel)
			mgoSelMap = make(map[string]bool)
			for _, s := range sel { mgoSelMap[s] = true }
		}
		for _, coll := range collections {
			if mgoSelMap != nil && !mgoSelMap[coll] { continue }
			fid := exportMongoToCSV(ctx, db, coll, conn, userID)
			if fid != "" { fileIDs = append(fileIDs, fid) }
		}

	case "snowflake":
		if conn.Host == "" { return nil }
		sfCfg := &sf.Config{
			Account:   conn.Host,
			User:      conn.Username,
			Password:  conn.Password,
			Database:  conn.Database,
			Warehouse: conn.Bucket,
		}
		sfDsn, err := sf.DSN(sfCfg)
		if err != nil { return nil }
		sfDB, err := sql.Open("snowflake", sfDsn)
		if err != nil { return nil }
		defer sfDB.Close()
		rows, _ := sfDB.Query("SHOW TABLES")
		if rows != nil {
			var tables []string
			for rows.Next() {
				var createdOn, name, kind, dbName, schemaName string
				rows.Scan(&createdOn, &name, &kind, &dbName, &schemaName)
				tables = append(tables, name)
			}
			rows.Close()
		var sfSelMap map[string]bool
		if conn.SelectedTables != "" {
			var sel []string
			json.Unmarshal([]byte(conn.SelectedTables), &sel)
			sfSelMap = make(map[string]bool)
			for _, s := range sel { sfSelMap[s] = true }
		}
			for _, table := range tables {
			if sfSelMap != nil && !sfSelMap[table] { continue }
				fid := exportTableToCSV(sfDB, table, conn, userID)
				if fid != "" { fileIDs = append(fileIDs, fid) }
			}
		}

	case "databricks":
		if conn.Endpoint == "" || conn.APIKey == "" { return nil }
		if conn.SelectedTables != "" {
			log.Printf("📊 Databricks refresh with selected tables: %s", conn.SelectedTables)
		}
		fid := fetchAPIToCSV(conn, userID)
		if fid != "" { fileIDs = append(fileIDs, fid) }

	case "google-drive", "google_drive":
		if conn.APIKey == "" { return nil }
		// List files and download CSVs
		client := &http.Client{Timeout: 30 * time.Second}
		req, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files?pageSize=50&q=mimeType%3D'text/csv'&fields=files(id,name)", nil)
		req.Header.Set("Authorization", "Bearer "+conn.APIKey)
		resp, err := client.Do(req)
		if err != nil { return nil }
		defer resp.Body.Close()
		var gResult struct { Files []struct { ID string `json:"id"`; Name string `json:"name"` } `json:"files"` }
		json.NewDecoder(resp.Body).Decode(&gResult)
		var gdSelMap map[string]bool
if conn.SelectedTables != "" {
var sel []string
json.Unmarshal([]byte(conn.SelectedTables), &sel)
gdSelMap = make(map[string]bool)
for _, s := range sel { gdSelMap[s] = true }
}
for _, f := range gResult.Files {
if gdSelMap != nil && !gdSelMap[f.Name] { continue }
			dlReq, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files/"+f.ID+"?alt=media", nil)
			dlReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
			dlResp, err := client.Do(dlReq)
			if err != nil { continue }
			fileID := uuid.New().String()[:16]
			filename := fmt.Sprintf("sync_gdrive_%s_%s", time.Now().Format("20060102"), f.Name)
			fpath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			file, _ := os.Create(fpath)
			written, _ := io.Copy(file, dlResp.Body)
			file.Close()
uploadToStorage(fpath, conn.UserID, "uploads")
			dlResp.Body.Close()
			if written > 0 {
				DB.Create(&UploadedFile{ID: fileID, Filename: filename, Path: fpath, Size: written, UserID: userID, CreatedAt: time.Now()})
				fileIDs = append(fileIDs, fileID)
			}
		}

	case "aws-s3", "aws_s3":
		// S3: download CSV/JSON files from bucket
		if conn.APIKey == "" || conn.Bucket == "" { return nil }
		sess, err := awssession.NewSession(&aws.Config{
			Region:      aws.String(conn.Region),
			Credentials: credentials.NewStaticCredentials(conn.APIKey, conn.Password, ""),
		})
		if err != nil { return nil }
		s3Client := s3.New(sess)
		result, err := s3Client.ListObjectsV2(&s3.ListObjectsV2Input{
			Bucket: aws.String(conn.Bucket), MaxKeys: aws.Int64(100),
		})
		if err != nil { return nil }
		for _, obj := range result.Contents {
			key := *obj.Key
			if conn.SelectedTables != "" {
				var s3Sel []string
				json.Unmarshal([]byte(conn.SelectedTables), &s3Sel)
				s3Map := make(map[string]bool)
				for _, s := range s3Sel { s3Map[s] = true }
				if !s3Map[key] { continue }
			}
			if strings.HasSuffix(key, ".csv") || strings.HasSuffix(key, ".json") {
				fid := downloadS3File(s3Client, conn.Bucket, key, userID)
				if fid != "" { fileIDs = append(fileIDs, fid) }
			}
		}

	case "rest_api", "rest":
		if conn.Endpoint == "" { return nil }
		if conn.RateLimitPaused {
			if conn.RateLimitResetAt != nil && time.Now().Before(*conn.RateLimitResetAt) {
				log.Printf("⏸️ Skipping refresh %s - rate limited until %s", conn.Name, conn.RateLimitResetAt.Format("15:04:05"))
				return nil
			}
			DB.Model(&conn).Updates(map[string]interface{}{"rate_limit_paused": false, "rate_limit": "Resumed"})
		}
		fid := fetchAPIToCSV(conn, userID)
		if fid != "" { fileIDs = append(fileIDs, fid) }

	case "graphql":
		if conn.Endpoint == "" { return nil }
		if conn.RateLimitPaused {
			if conn.RateLimitResetAt != nil && time.Now().Before(*conn.RateLimitResetAt) {
				log.Printf("⏸️ Skipping refresh %s - rate limited until %s", conn.Name, conn.RateLimitResetAt.Format("15:04:05"))
				return nil
			}
			DB.Model(&conn).Updates(map[string]interface{}{"rate_limit_paused": false, "rate_limit": "Resumed"})
		}
		fid := fetchGraphQLToCSV(conn, userID)
		if fid != "" { fileIDs = append(fileIDs, fid) }

	case "pinecone", "weaviate", "chroma", "lancedb":
		fid := fetchVectorDBToCSV(conn, userID)
		if fid != "" { fileIDs = append(fileIDs, fid) }

	case "gcs":
		// GCS: similar to S3 but via REST API
		if conn.APIKey == "" || conn.Bucket == "" { return nil }
		fid := fetchGCSToCSV(conn, userID)
		if fid != "" { fileIDs = append(fileIDs, fid) }
	}

	return fileIDs
}

// exportTableToCSV - exports a SQL table to CSV file
func exportTableToCSV(sqlDB *sql.DB, table string, conn Connection, userID string) string {
	rows, err := sqlDB.Query(fmt.Sprintf("SELECT * FROM %s LIMIT 50000", sanitizeTableName(table)))
	if err != nil { return "" }
	defer rows.Close()

	columns, _ := rows.Columns()
	if len(columns) == 0 { return "" }

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_%s_%s_%s.csv", conn.Database, table, time.Now().Format("20060102_150405"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)

	file, err := os.Create(filepath)
	if err != nil { return "" }
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Write(columns)

	rowCount := 0
	values := make([]interface{}, len(columns))
	valuePtrs := make([]interface{}, len(columns))
	for i := range values { valuePtrs[i] = &values[i] }

	for rows.Next() {
		rows.Scan(valuePtrs...)
		row := make([]string, len(columns))
		for i, v := range values {
			if v == nil { row[i] = "" } else { row[i] = fmt.Sprintf("%v", v) }
		}
		writer.Write(row)
		rowCount++
	}
	writer.Flush()

	if rowCount == 0 { os.Remove(filepath); return "" }

	fileInfo, _ := os.Stat(filepath)
fileSizeMB := float64(fileInfo.Size()) / (1024 * 1024)
if ok, _ := CheckStorage(userID, fileSizeMB); !ok {
os.Remove(filepath)
log.Printf("⚠️ Storage limit exceeded, skipping %s (%.1f MB)", filepath, fileSizeMB)
return ""
}
	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: fileInfo.Size(), UserID: userID, CreatedAt: time.Now(),
	})
	log.Printf("📄 Exported %s.%s: %d rows → %s", conn.Database, table, rowCount, filename)
	return fileID
}

// exportMongoToCSV - exports a MongoDB collection to CSV
func exportMongoToCSV(ctx context.Context, db *mongo.Database, collection string, conn Connection, userID string) string {
	coll := db.Collection(collection)
	cursor, err := coll.Find(ctx, bson.M{}, options.Find().SetLimit(50000))
	if err != nil { return "" }
	defer cursor.Close(ctx)

	var results []bson.M
	if err := cursor.All(ctx, &results); err != nil || len(results) == 0 { return "" }

	// Collect all keys
	keySet := make(map[string]bool)
	for _, doc := range results {
		for k := range doc { keySet[k] = true }
	}
	var headers []string
	for k := range keySet { headers = append(headers, k) }

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_%s_%s_%s.csv", conn.Database, collection, time.Now().Format("20060102_150405"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)

	file, err := os.Create(filepath)
	if err != nil { return "" }
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Write(headers)
	for _, doc := range results {
		row := make([]string, len(headers))
		for i, h := range headers {
			if v, ok := doc[h]; ok { row[i] = fmt.Sprintf("%v", v) } else { row[i] = "" }
		}
		writer.Write(row)
	}
	writer.Flush()

	fileInfo, _ := os.Stat(filepath)
fileSizeMB := float64(fileInfo.Size()) / (1024 * 1024)
if ok, _ := CheckStorage(userID, fileSizeMB); !ok {
os.Remove(filepath)
log.Printf("⚠️ Storage limit exceeded, skipping %s (%.1f MB)", filepath, fileSizeMB)
return ""
}
	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: fileInfo.Size(), UserID: userID, CreatedAt: time.Now(),
	})
	log.Printf("📄 Exported mongo %s.%s: %d docs → %s", conn.Database, collection, len(results), filename)
	return fileID
}

// downloadS3File - downloads a file from S3
func downloadS3File(s3Client *s3.S3, bucket, key, userID string) string {
	result, err := s3Client.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(bucket), Key: aws.String(key),
	})
	if err != nil { return "" }
	defer result.Body.Close()

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_s3_%s_%s", time.Now().Format("20060102_150405"), strings.ReplaceAll(key, "/", "_"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)

	file, err := os.Create(filepath)
	if err != nil { return "" }
	defer file.Close()

	written, _ := io.Copy(file, result.Body)
	if written == 0 { os.Remove(filepath); return "" }

	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: written, UserID: userID, CreatedAt: time.Now(),
	})
	log.Printf("📄 Downloaded S3 %s/%s: %d bytes", bucket, key, written)
	return fileID
}

// fetchAPIToCSV - fetches REST API data and saves as CSV

// parseRateLimitReset parses reset time from response headers
func parseRateLimitReset(resp *http.Response) *time.Time {
	reset := resp.Header.Get("X-RateLimit-Reset")
	if reset == "" { reset = resp.Header.Get("RateLimit-Reset") }
	if reset == "" { reset = resp.Header.Get("Retry-After") }
	if reset != "" {
		// Try unix timestamp
		if ts, err := strconv.ParseInt(reset, 10, 64); err == nil {
			t := time.Unix(ts, 0)
			return &t
		}
		// Try seconds
		if secs, err := strconv.Atoi(reset); err == nil {
			t := time.Now().Add(time.Duration(secs) * time.Second)
			return &t
		}
	}
	// Default: 1 hour from now
	t := time.Now().Add(1 * time.Hour)
	return &t
}

func fetchAPIToCSV(conn Connection, userID string) string {
	// Check if rate limit paused - skip until reset time
	if conn.RateLimitPaused {
		if conn.RateLimitResetAt != nil && time.Now().Before(*conn.RateLimitResetAt) {
			log.Printf("⏸️ Skipping %s - rate limited until %s", conn.Name, conn.RateLimitResetAt.Format("15:04:05"))
			return ""
		}
		// Reset time passed, unpause
		log.Printf("▶️ Rate limit reset, resuming: %s", conn.Name)
		DB.Model(&conn).Updates(map[string]interface{}{
			"rate_limit_paused": false, "rate_limit": "Resumed",
			"rate_limit_remaining": conn.RateLimitDaily,
		})
		conn.RateLimitPaused = false
	}

	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", conn.Endpoint, nil)
	if conn.APIKey != "" { req.Header.Set("Authorization", "Bearer "+conn.APIKey) }
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()

	now := time.Now()
	DB.Model(&conn).Updates(map[string]interface{}{"api_calls_count": gorm.Expr("api_calls_count + 1"), "last_poll_at": now})

	if resp.StatusCode == 429 {
		log.Printf("⚠️ Rate limited: %s - pausing polling", conn.Name)
		resetAt := parseRateLimitReset(resp)
		DB.Model(&conn).Updates(map[string]interface{}{
			"rate_limit": "Rate limited - paused",
			"rate_limit_paused": true,
			"rate_limit_remaining": 0,
			"rate_limit_reset_at": resetAt,
		})
		// Log to usage
		DB.Create(&UsageLog{
			ID: fmt.Sprintf("sync-%s-%d", conn.ID[:8], now.Unix()),
			UserID: userID, EventType: "sync", EventName: "API Rate Limited: " + conn.Name,
			ResourceID: conn.ID, ResourceName: conn.Name,
			CreditsUsed: 0, CreatedAt: now,
		})
		return ""
	}

	// Update rate limit from headers
	rlLimit := resp.Header.Get("X-RateLimit-Limit")
	if rlLimit == "" { rlLimit = resp.Header.Get("RateLimit-Limit") }
	rlRemaining := resp.Header.Get("X-RateLimit-Remaining")
	if rlRemaining == "" { rlRemaining = resp.Header.Get("RateLimit-Remaining") }
	rlReset := resp.Header.Get("X-RateLimit-Reset")
	if rlReset == "" { rlReset = resp.Header.Get("RateLimit-Reset") }

	if rlLimit != "" {
		daily, _ := strconv.Atoi(rlLimit)
		remaining, _ := strconv.Atoi(rlRemaining)
		rlStr := rlRemaining + "/" + rlLimit
		updates := map[string]interface{}{
			"rate_limit": rlStr,
			"rate_limit_daily": daily,
			"rate_limit_remaining": remaining,
		}
		if rlReset != "" {
			if resetUnix, err := strconv.ParseInt(rlReset, 10, 64); err == nil {
				t := time.Unix(resetUnix, 0)
				updates["rate_limit_reset_at"] = t
			}
		}
		// Auto-pause if remaining < 5% of daily
		if daily > 0 && remaining > 0 && remaining < daily/20 {
			log.Printf("⚠️ Rate limit low (%d/%d), pausing: %s", remaining, daily, conn.Name)
			updates["rate_limit_paused"] = true
			updates["rate_limit"] = fmt.Sprintf("%d/%d - paused (low)", remaining, daily)
		}
		DB.Model(&conn).Updates(updates)
	}

	// Log sync event to usage
	DB.Create(&UsageLog{
		ID: fmt.Sprintf("sync-%s-%d", conn.ID[:8], now.Unix()),
		UserID: userID, EventType: "sync", EventName: "Data Sync: " + conn.Name,
		ResourceID: conn.ID, ResourceName: conn.Name,
		CreditsUsed: 0.001, CreatedAt: now,
	})

	body, _ := io.ReadAll(resp.Body)

	// Try to parse as JSON array
	var records []map[string]interface{}
	if err := json.Unmarshal(body, &records); err != nil {
		// Try wrapped: {"data": [...]}
		var wrapped map[string]interface{}
		if err := json.Unmarshal(body, &wrapped); err != nil { return "" }
		for _, v := range wrapped {
			if arr, ok := v.([]interface{}); ok {
				for _, item := range arr {
					if m, ok := item.(map[string]interface{}); ok { records = append(records, m) }
				}
				break
			}
		}
	}
	if len(records) == 0 { return "" }

	// Collect headers
	keySet := make(map[string]bool)
	for _, r := range records { for k := range r { keySet[k] = true } }
	var headers []string
	for k := range keySet { headers = append(headers, k) }

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_api_%s_%s.csv", sanitizeFilename(conn.Name), time.Now().Format("20060102_150405"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)

	file, _ := os.Create(filepath)
	defer file.Close()
	writer := csv.NewWriter(file)
	writer.Write(headers)
	for _, r := range records {
		row := make([]string, len(headers))
		for i, h := range headers {
			if v, ok := r[h]; ok { row[i] = fmt.Sprintf("%v", v) } else { row[i] = "" }
		}
		writer.Write(row)
	}
	writer.Flush()

	fileInfo, _ := os.Stat(filepath)
fileSizeMB := float64(fileInfo.Size()) / (1024 * 1024)
if ok, _ := CheckStorage(userID, fileSizeMB); !ok {
os.Remove(filepath)
log.Printf("⚠️ Storage limit exceeded, skipping %s (%.1f MB)", filepath, fileSizeMB)
return ""
}
	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: fileInfo.Size(), UserID: userID, CreatedAt: time.Now(),
	})
	log.Printf("📄 Fetched API %s: %d records → %s", conn.Name, len(records), filename)
	return fileID
}

// fetchGraphQLToCSV - fetches GraphQL data
func fetchGraphQLToCSV(conn Connection, userID string) string {
	// Execute a generic query to get data
	body := `{"query":"{ __schema { queryType { fields { name } } } }"}`
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("POST", conn.Endpoint, bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" { req.Header.Set("Authorization", "Bearer "+conn.APIKey) }
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_graphql_%s_%s.json", sanitizeFilename(conn.Name), time.Now().Format("20060102_150405"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
	os.WriteFile(filepath, respBody, 0644)

	fileInfo, _ := os.Stat(filepath)
fileSizeMB := float64(fileInfo.Size()) / (1024 * 1024)
if ok, _ := CheckStorage(userID, fileSizeMB); !ok {
os.Remove(filepath)
log.Printf("⚠️ Storage limit exceeded, skipping %s (%.1f MB)", filepath, fileSizeMB)
return ""
}
	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: fileInfo.Size(), UserID: userID, CreatedAt: time.Now(),
	})
	return fileID
}

// fetchVectorDBToCSV - fetches vector DB data
func fetchVectorDBToCSV(conn Connection, userID string) string {
	if conn.SelectedTables != "" {
		log.Printf("📊 VectorDB fetch with selected: %s", conn.SelectedTables)
	}
	var url string
	var headerKey, headerVal string

	switch conn.SubType {
	case "pinecone":
		url = conn.Endpoint + "/describe_index_stats"
		headerKey = "Api-Key"
		headerVal = conn.APIKey
	case "weaviate":
		url = conn.Endpoint + "/v1/objects?limit=1000"
		headerKey = "Authorization"
		headerVal = "Bearer " + conn.APIKey
	case "chroma":
		url = conn.Endpoint + "/api/v1/collections"
		headerKey = "Authorization"
		headerVal = "Bearer " + conn.APIKey
	case "lancedb":
		url = conn.Endpoint + "/v1/table"
		headerKey = "Authorization"
		headerVal = "Bearer " + conn.APIKey
	default:
		return ""
	}

	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	if headerVal != "" { req.Header.Set(headerKey, headerVal) }
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	fileID := uuid.New().String()[:16]
	filename := fmt.Sprintf("sync_%s_%s_%s.json", conn.SubType, sanitizeFilename(conn.Name), time.Now().Format("20060102_150405"))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
	os.WriteFile(filepath, body, 0644)

	fileInfo, _ := os.Stat(filepath)
fileSizeMB := float64(fileInfo.Size()) / (1024 * 1024)
if ok, _ := CheckStorage(userID, fileSizeMB); !ok {
os.Remove(filepath)
log.Printf("⚠️ Storage limit exceeded, skipping %s (%.1f MB)", filepath, fileSizeMB)
return ""
}
	DB.Create(&UploadedFile{
		ID: fileID, Filename: filename, Path: filepath,
		Size: fileInfo.Size(), UserID: userID, CreatedAt: time.Now(),
	})
	return fileID
}

// fetchGCSToCSV - fetches files from Google Cloud Storage
func fetchGCSToCSV(conn Connection, userID string) string {
	url := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?maxResults=100", conn.Bucket)
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+conn.APIKey)
	resp, err := client.Do(req)
	if err != nil { return "" }
	defer resp.Body.Close()

	var result struct {
		Items []struct {
			Name string `json:"name"`
		} `json:"items"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	var fileIDs []string
	var gcsSelMap map[string]bool
	if conn.SelectedTables != "" {
		var sel []string
		json.Unmarshal([]byte(conn.SelectedTables), &sel)
		gcsSelMap = make(map[string]bool)
		for _, s := range sel { gcsSelMap[s] = true }
	}
	for _, item := range result.Items {
		if !strings.HasSuffix(item.Name, ".csv") && !strings.HasSuffix(item.Name, ".json") { continue }
		if gcsSelMap != nil && !gcsSelMap[item.Name] { continue }
		objURL := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media", conn.Bucket, item.Name)
		req2, _ := http.NewRequest("GET", objURL, nil)
		req2.Header.Set("Authorization", "Bearer "+conn.APIKey)
		resp2, err := client.Do(req2)
		if err != nil { continue }

		fileID := uuid.New().String()[:16]
		filename := fmt.Sprintf("sync_gcs_%s_%s", time.Now().Format("20060102"), strings.ReplaceAll(item.Name, "/", "_"))
		filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
		file, _ := os.Create(filepath)
		written, _ := io.Copy(file, resp2.Body)
		file.Close()
		resp2.Body.Close()

		if written > 0 {
			DB.Create(&UploadedFile{
				ID: fileID, Filename: filename, Path: filepath,
				Size: written, UserID: userID, CreatedAt: time.Now(),
			})
			fileIDs = append(fileIDs, fileID)
		}
	}

	if len(fileIDs) > 0 { return fileIDs[0] }
	return ""
}

// === HTTP HANDLERS ===

func UpdateModelSyncHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut && r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	var req struct {
		ModelID       string `json:"model_id"`
		SyncMode      string `json:"sync_mode"`
		ScheduleCron  string `json:"schedule_cron"`
		ScheduleDesc  string `json:"schedule_desc"`
		ConnectionIDs string `json:"connection_ids"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	var model FineTunedModel
	if DB.Where("id = ? AND user_id = ?", req.ModelID, userID).First(&model).Error != nil {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	updates := map[string]interface{}{"sync_mode": req.SyncMode}

	switch req.SyncMode {
	case "scheduled":
		updates["schedule_cron"] = req.ScheduleCron
		updates["schedule_desc"] = req.ScheduleDesc
		nextRun := calculateNextRun(req.ScheduleCron)
		updates["next_sync_at"] = nextRun
		GlobalWatcher.StopWatching(model.ID)
		model.ScheduleCron = req.ScheduleCron
		model.SyncMode = "scheduled"
		GlobalScheduler.AddJob(model)

	case "real-time":
		updates["connection_ids"] = req.ConnectionIDs
		GlobalScheduler.RemoveJob(model.ID)
		model.ConnectionIDs = req.ConnectionIDs
		model.SyncMode = "real-time"
		GlobalWatcher.StartWatching(model)

	case "manual":
		GlobalScheduler.RemoveJob(model.ID)
		GlobalWatcher.StopWatching(model.ID)
		updates["schedule_cron"] = ""
		updates["connection_ids"] = ""
		updates["next_sync_at"] = nil
	}

	DB.Model(&FineTunedModel{}).Where("id = ?", req.ModelID).Updates(updates)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "sync_mode": req.SyncMode})
}

func TriggerSchedulerSyncHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Tüm real-time ve scheduled modelleri sync et
	var models []FineTunedModel
	DB.Where("sync_mode IN ? AND status = ?", []string{"real-time", "scheduled"}, "active").Find(&models)

	type SyncedModel struct {
		ID      string   `json:"id"`
		Name    string   `json:"name"`
		UserID  string   `json:"user_id"`
		FileIDs []string `json:"file_ids"`
	}

	var synced []SyncedModel
	for _, m := range models {
		fileIDs := []string{}
		if m.SourceFiles != "" {
			for _, fid := range strings.Split(m.SourceFiles, ",") {
				fid = strings.TrimSpace(fid)
				if fid != "" {
					fileIDs = append(fileIDs, fid)
				}
			}
		}
		synced = append(synced, SyncedModel{
			ID:      m.ID,
			Name:    m.Name,
			UserID:  m.UserID,
			FileIDs: fileIDs,
		})
		log.Printf("[SCHEDULER SYNC] Model: %s", m.Name)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "ok",
		"synced_models": synced,
		"count":         len(synced),
	})
}

func GetSchedulerStatusHandler(w http.ResponseWriter, r *http.Request) {
	GlobalScheduler.mu.Lock()
	defer GlobalScheduler.mu.Unlock()

	jobs := make([]map[string]interface{}, 0)
	for _, job := range GlobalScheduler.jobs {
		j := map[string]interface{}{
			"model_id": job.ModelID, "model_name": job.ModelName,
			"cron": job.CronExpr, "next_run": job.NextRun, "status": job.Status,
		}
		if job.LastRun != nil { j["last_run"] = job.LastRun }
		jobs = append(jobs, j)
	}

	// Real-time watchers
	GlobalWatcher.mu.Lock()
	watchers := make([]map[string]string, 0)
	for key, w := range GlobalWatcher.watchers {
		watchers = append(watchers, map[string]string{
			"key": key, "model_id": w.ModelID,
			"connection_id": w.ConnectionID, "type": w.ConnType,
		})
	}
	GlobalWatcher.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"scheduled_jobs": jobs, "realtime_watchers": watchers,
		"total_scheduled": len(jobs), "total_realtime": len(watchers),
	})
}
