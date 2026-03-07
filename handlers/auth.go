package handlers

import (
	"bytes"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
"log"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
"strings"
	"time"

	"sync"

	"github.com/go-redis/redis/v8"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/mysql"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	awssession "github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	sf "github.com/snowflakedb/gosnowflake"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/drive/v3"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// sanitizeTableName - SQL injection önlemek için table name validate et
func sanitizeTableName(name string) string {
	// Sadece alphanumeric ve underscore izin ver
	cleaned := ""
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' {
			cleaned += string(c)
		}
	}
	if cleaned == "" {
		return "invalid_table"
	}
	return cleaned
}

var DB *gorm.DB
var rdb *redis.Client
var ctx = context.Background()

type User struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	Name      string    `json:"name"`
	Email     string    `gorm:"unique" json:"email"`
	Password  string    `json:"-"`
	Image     string    `json:"image"`
	Role      string    `json:"role"`
	Plan      string    `json:"plan"`
	MaxTeams  int       `json:"max_teams"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Session struct {
	UserID    string    `json:"user_id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	ExpiresAt time.Time `json:"expires_at"`
}

// Email Verification Code
type VerificationCode struct {
	ID        uint      `gorm:"primaryKey" json:"id"`
	Email     string    `gorm:"index" json:"email"`
	Code      string    `json:"code"`
	ExpiresAt time.Time `json:"expires_at"`
	Used      bool      `gorm:"default:false" json:"used"`
	CreatedAt time.Time `json:"created_at"`
}

// Password Reset Token
type PasswordResetToken struct {
	ID        uint      `gorm:"primaryKey" json:"id"`
	UserID    string    `gorm:"index" json:"user_id"`
	Token     string    `gorm:"uniqueIndex" json:"token"`
	ExpiresAt time.Time `json:"expires_at"`
	Used      bool      `gorm:"default:false" json:"used"`
	CreatedAt time.Time `json:"created_at"`
}

func InitAuth() error {
	// PostgreSQL
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = GetDatabaseURL()
	}

	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		return err
	}

	// AutoMigrate in background to not block startup
	go func() {
		DB.AutoMigrate(&User{}, &UploadedFile{}, &Query{}, &Message{}, &QueryFile{}, &FineTunedModel{}, &Folder{}, &Connection{}, &APIKey{}, &VerificationCode{}, &PasswordResetToken{}, &UserQuota{}, &VerticalConfig{}, &VerticalTool{}, &VerticalAgent{}, &Endpoint{}, &PredictionStore{}, &LLMSecret{})
		// Create indexes for performance
		DB.Exec("CREATE INDEX IF NOT EXISTS idx_queries_user_updated ON queries(user_id, updated_at DESC)")
		DB.Exec("CREATE INDEX IF NOT EXISTS idx_uploaded_files_user ON uploaded_files(user_id, created_at DESC)")
		DB.Exec("CREATE INDEX IF NOT EXISTS idx_fine_tuned_models_user ON fine_tuned_models(user_id, created_at DESC)")
		DB.Exec("CREATE INDEX IF NOT EXISTS idx_connections_user ON connections(user_id)")
DB.Exec("CREATE INDEX IF NOT EXISTS idx_messages_query ON messages(query_id, created_at)")
DB.Exec("CREATE INDEX IF NOT EXISTS idx_fine_tuned_models_status ON fine_tuned_models(user_id, status)")
DB.Exec("CREATE INDEX IF NOT EXISTS idx_queries_model ON queries(user_id, training_model_id)")
	}()

	// Redis
	redisURL := GetRedisURL()
	rdb = redis.NewClient(&redis.Options{
		Addr:         redisURL,
		Password:     os.Getenv("REDIS_PASSWORD"),
		DialTimeout:  3 * time.Second,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
	})

	return rdb.Ping(ctx).Err()
}

func generateSessionID() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func CreateSession(userID, email, name string) (string, error) {
	sessionID := generateSessionID()
	session := Session{
		UserID:    userID,
		Email:     email,
		Name:      name,
		ExpiresAt: time.Now().Add(7 * 24 * time.Hour),
	}

	data, _ := json.Marshal(session)
	err := rdb.Set(ctx, "session:"+sessionID, data, 7*24*time.Hour).Err()
	return sessionID, err
}

func GetSession(sessionID string) (*Session, error) {
	data, err := rdb.Get(ctx, "session:"+sessionID).Result()
	if err != nil {
		return nil, err
	}

	var session Session
	json.Unmarshal([]byte(data), &session)
	return &session, nil
}

func DeleteSession(sessionID string) error {
	return rdb.Del(ctx, "session:"+sessionID).Err()
}

func SignupHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Name     string `json:"name"`
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	if req.Email == "" || req.Password == "" {
		http.Error(w, "Email and password required", http.StatusBadRequest)
		return
	}

	// Check if exists
	var existing User
	if DB.Where("email = ?", req.Email).First(&existing).Error == nil {
		http.Error(w, "Email already exists", http.StatusBadRequest)
		return
	}

	// Hash password
	hashed, _ := bcrypt.GenerateFromPassword([]byte(req.Password), 12)

	user := User{
		ID:        generateSessionID()[:24],
		Name:      req.Name,
		Email:     req.Email,
		Password:  string(hashed),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := DB.Create(&user).Error; err != nil {
		http.Error(w, "Failed to create user", http.StatusInternalServerError)
		return
	}

	// Create session
	sessionID, _ := CreateSession(user.ID, user.Email, user.Name)
	AuthEventsTotal.WithLabelValues("register").Inc()
	go SendNewUserNotification(user.Name, user.Email)

	http.SetCookie(w, &http.Cookie{
		Name:     "session",
		Value:    sessionID,
		Path:     "/",
		HttpOnly: true,
		MaxAge:   7 * 24 * 60 * 60,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"user": user,
	})
}

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	var user User
	if DB.Where("email = ?", req.Email).First(&user).Error != nil {
		AuthEventsTotal.WithLabelValues("login_failed").Inc()
	http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	if bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(req.Password)) != nil {
		AuthEventsTotal.WithLabelValues("login_failed").Inc()
	http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	sessionID, _ := CreateSession(user.ID, user.Email, user.Name)
	AuthEventsTotal.WithLabelValues("login").Inc()

	http.SetCookie(w, &http.Cookie{
		Name:     "session",
		Value:    sessionID,
		Path:     "/",
		HttpOnly: true,
		MaxAge:   7 * 24 * 60 * 60,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"user": user,
	})
}

func LogoutHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err == nil {
		DeleteSession(cookie.Value)
	}

	http.SetCookie(w, &http.Cookie{
		Name:   "session",
		Value:  "",
		Path:   "/",
		MaxAge: -1,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "logged out"})
}

func MeHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	session, err := GetSession(cookie.Value)
	if err != nil {
		http.Error(w, "Invalid session", http.StatusUnauthorized)
		return
	}

	var user User
	DB.Where("id = ?", session.UserID).First(&user)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
				cookie, err := r.Cookie("session")
		if err != nil {
			http.Error(w, "Not authenticated", http.StatusUnauthorized)
			return
		}

		session, err := GetSession(cookie.Value)
		if err != nil {
			http.Error(w, "Invalid session", http.StatusUnauthorized)
			return
		}

		r.Header.Set("X-User-ID", session.UserID)
		next(w, r)
	}
}

// Query model
type Query struct {
	ID              string    `gorm:"primaryKey" json:"id"`
	Name            string    `json:"name"`
	Model           string    `json:"model"`
	UserID          string    `json:"user_id"`
	IsTraining      bool      `json:"is_training"`
	HasModel        bool      `json:"has_model"`
	TrainingFailed  bool      `gorm:"column:training_failed" json:"trainingFailed"`
	TrainingModelID *string   `json:"training_model_id"`
	CreatedAt       time.Time `json:"created_at"`
	FileID          string    `json:"file_id"`
	UpdatedAt       time.Time `json:"updated_at"`
	ModelName       string    `gorm:"column:model_name" json:"modelName"`
	ModelAccuracy   float64   `gorm:"column:model_accuracy" json:"modelAccuracy"`
	SourceCsvName   string    `gorm:"column:source_csv_name" json:"sourceCsvName"`
	Source          string    `gorm:"column:source;default:playground" json:"source"`
}

func (Query) TableName() string { return "queries" }

// Message model
type Message struct {
	ID               string    `gorm:"primaryKey" json:"id"`
	Role             string    `json:"role"`
	Content          string    `gorm:"type:text" json:"content"`
	Model            string    `json:"model"`
	Tokens           int       `json:"tokens"`
	QueryID          string    `json:"query_id"`
	UserID           string    `json:"user_id"`
	CreatedAt        time.Time `json:"created_at"`
	FineTunedModelID string    `json:"finetuned_model_id"`
	CompareGroup     string    `json:"compare_group"`
	TimeTaken        string    `json:"time_taken"`
	FunctionCalls    string    `json:"function_calls" gorm:"type:jsonb;default:null"`
}

// QueryFile - many to many
type QueryFile struct {
	QueryID string `gorm:"primaryKey" json:"query_id"`
	FileID  string `gorm:"primaryKey" json:"file_id"`
}

type FineTunedModel struct {
	ID               string     `gorm:"primaryKey" json:"id"`
	Name             string     `json:"name"`
	Version          int        `json:"version"`
	SourceFileID     string     `json:"source_file_id"`
	SourceName       string     `json:"source_name"`
	SourceFiles      string     `json:"source_files"`
	ModelPath        string     `json:"model_path"`
	Accuracy         float64    `json:"accuracy"`
	Epochs           int        `json:"epochs"`
	BatchSize        int        `json:"batch_size"`
	Loss             float64    `json:"loss"`
	TrainingDuration int        `json:"training_duration"`
	UserID           string     `json:"user_id"`
	CreatedAt        time.Time  `json:"created_at"`
	SyncMode         string     `json:"sync_mode" gorm:"default:manual"`
	ScheduleCron     string     `json:"schedule_cron"`
	ScheduleDesc     string     `json:"schedule_desc"`
	LastSyncAt       *time.Time `json:"last_sync_at"`
	NextSyncAt       *time.Time `json:"next_sync_at"`
	SyncStatus       string     `json:"sync_status" gorm:"default:idle"`
	ConnectionIDs    string     `json:"connection_ids"`
Status           string     `json:"status" gorm:"default:active"`
TrainingEpoch    int        `json:"training_epoch" gorm:"default:0"`
TrainingLoss     float64    `json:"training_loss" gorm:"default:0"`
TrainingAcc      float64    `json:"training_acc" gorm:"default:0"`
}

type Folder struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	Name      string    `json:"name"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
}

// Connection types
type Connection struct {
	ID             string     `gorm:"primaryKey" json:"id"`
	Name           string     `json:"name"`
	Type           string     `json:"type"`     // database, vectordb, cloud, api
	SubType        string     `json:"sub_type"` // postgresql, mysql, pinecone, etc.
	Host           string     `json:"host"`
	Port           int        `json:"port"`
	Database       string     `json:"database"`
	Username       string     `json:"username"`
	Password       string     `json:"-"`
	APIKey         string     `json:"-"`
	Endpoint       string     `json:"endpoint"`
	Bucket         string     `json:"bucket"`
	Region         string     `json:"region"`
	SSL            bool       `json:"ssl"`
	Status         string     `json:"status"` // active, error, disconnected
	LastTestedAt   *time.Time `json:"last_tested_at"`
	CachedTables   string     `json:"cached_tables"`
	CachedAt       *time.Time `json:"cached_at"`
	SelectedTables string     `json:"selected_tables"`
RateLimit         string     `json:"rate_limit"`
	RateLimitDaily    int        `json:"rate_limit_daily"`
	RateLimitRemaining int       `json:"rate_limit_remaining"`
	RateLimitResetAt  *time.Time `json:"rate_limit_reset_at"`
	RateLimitPaused   bool       `json:"rate_limit_paused"`
	APICallsCount     int        `json:"api_calls_count"`
	LastPollAt        *time.Time `json:"last_poll_at"`
	UserID            string     `json:"user_id"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
}

type APIKey struct {
	ID             string     `gorm:"primaryKey" json:"id"`
	Name           string     `json:"name"`
	Key            string     `json:"key"`
	KeyHash        string     `json:"-"`
	UserID         string     `json:"user_id"`
	Permissions    string     `json:"permissions"`
	RateLimit      string     `json:"rate_limit"`
	Requests       int        `json:"requests"`
	FineTunedModel string     `gorm:"column:finetuned_model" json:"finetuned_model"`
	LLMProvider    string     `json:"llm_provider"`
	LLMModel       string     `json:"llm_model"`
	LastUsed       *time.Time `json:"last_used"`
	CreatedAt      time.Time  `json:"created_at"`
}

func CreateConnectionHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, err := GetSession(cookie.Value)
	if err != nil || session == nil {
		http.Error(w, "Invalid session", http.StatusUnauthorized)
		return
	}

// Check storage quota
if ok, reason := CheckStorage(session.UserID, 10); !ok {
http.Error(w, reason, http.StatusForbidden)
return
}

	var input struct {
		Name     string `json:"name"`
		Type     string `json:"type"`
		SubType  string `json:"sub_type"`
		Host     string `json:"host"`
		Port     int    `json:"port"`
		Database string `json:"database"`
		Username string `json:"username"`
		Password string `json:"password"`
		APIKey   string `json:"api_key"`
		Endpoint string `json:"endpoint"`
		Bucket   string `json:"bucket"`
		Region   string `json:"region"`
		SSL      bool   `json:"ssl"`
	}

	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	conn := Connection{
		ID:       generateSessionID()[:16],
		Name:     input.Name,
		Type:     input.Type,
		SubType:  input.SubType,
		Host:     input.Host,
		Port:     input.Port,
		Database: input.Database,
		Username: input.Username,
		Password: input.Password,
		APIKey:   input.APIKey,
		Endpoint: input.Endpoint,
		Bucket:   input.Bucket,
		Region:   input.Region,
		SSL:      input.SSL,
		Status:   "active",
		UserID:   session.UserID,
	}

if err := DB.Create(&conn).Error; err != nil {
http.Error(w, err.Error(), http.StatusInternalServerError)
return
}

// Log connection creation to usage
DB.Create(&UsageLog{
ID: fmt.Sprintf("conn-%s", conn.ID[:8]),
UserID: session.UserID, EventType: "connection",
EventName: "Connection Created: " + conn.Name,
ResourceID: conn.ID, ResourceName: conn.Name,
CreditsUsed: 0, CreatedAt: time.Now(),
})

// Auto-detect rate limit for REST API on create
if (conn.SubType == "rest_api" || conn.SubType == "rest") && conn.Endpoint != "" {
client2 := &http.Client{Timeout: 10 * time.Second}
req2, _ := http.NewRequest("GET", conn.Endpoint, nil)
if conn.APIKey != "" { req2.Header.Set("Authorization", "Bearer "+conn.APIKey) }
resp2, err2 := client2.Do(req2)
if err2 == nil {
rl := resp2.Header.Get("X-RateLimit-Limit")
rem := resp2.Header.Get("X-RateLimit-Remaining")
if rl != "" {
s := rl + " req limit"
if rem != "" { s = rem + "/" + rl }
DB.Model(&Connection{}).Where("id = ?", conn.ID).Update("rate_limit", s)
} else if resp2.StatusCode == 200 {
body2, _ := io.ReadAll(io.LimitReader(resp2.Body, 10*1024*1024))
var arr []interface{}
if json.Unmarshal(body2, &arr) == nil {
rc := len(arr)
if rc == 1000 || rc == 500 || rc == 100 || rc == 10000 || rc == 5000 || rc == 2000 || rc == 50 || rc == 25 || rc == 200 {
DB.Model(&Connection{}).Where("id = ?", conn.ID).Update("rate_limit", fmt.Sprintf("%d rows/request (API default limit)", rc))
}
}
}
if resp2.StatusCode == 429 {
DB.Model(&Connection{}).Where("id = ?", conn.ID).Update("rate_limit", "Rate limited")
}
resp2.Body.Close()
}
}

json.NewEncoder(w).Encode(map[string]interface{}{
"id":       conn.ID,
"name":     conn.Name,
"type":     conn.Type,
"sub_type": conn.SubType,
"status":   conn.Status,
})
}

func ListConnectionsHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	var connections []Connection
	DB.Where("user_id = ?", session.UserID).Order("created_at DESC").Find(&connections)

	// Don't expose passwords/api keys
	result := make([]map[string]interface{}, len(connections))
	for i, c := range connections {
		result[i] = map[string]interface{}{
			"id":         c.ID,
			"name":       c.Name,
			"type":       c.Type,
			"sub_type":   c.SubType,
			"host":       c.Host,
			"port":       c.Port,
			"database":   c.Database,
			"username":   c.Username,
			"endpoint":   c.Endpoint,
			"bucket":     c.Bucket,
			"status":     c.Status,
"rate_limit": c.RateLimit,
			"rate_limit_daily": c.RateLimitDaily,
			"rate_limit_remaining": c.RateLimitRemaining,
			"rate_limit_reset_at": c.RateLimitResetAt,
			"rate_limit_paused": c.RateLimitPaused,
			"api_calls_count": c.APICallsCount,
			"last_poll_at": c.LastPollAt,
			"created_at": c.CreatedAt,
		}
		// Add cached rows/cols
		if c.CachedTables != "" {
			var cached struct {
				TableDetails []struct {
					Name    string `json:"name"`
					Rows    int64  `json:"rows"`
					Columns int    `json:"columns"`
				} `json:"table_details"`
			}
			json.Unmarshal([]byte(c.CachedTables), &cached)
			var totalRows int64
			var totalCols int
			var schemaNames []string
			for _, t := range cached.TableDetails {
				totalRows += t.Rows
				totalCols += t.Columns
				schemaNames = append(schemaNames, t.Name)
			}
			result[i]["total_rows"] = totalRows
			result[i]["total_cols"] = totalCols
			result[i]["schema"] = schemaNames
result[i]["table_details"] = cached.TableDetails
		}
	}

	json.NewEncoder(w).Encode(map[string]interface{}{"connections": result})
}

func DeleteConnectionHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Missing id", http.StatusBadRequest)
		return
	}

	DB.Where("id = ? AND user_id = ?", id, session.UserID).Delete(&Connection{}, &APIKey{})

	// Delete associated uploaded files (connection exports with conn_ prefix)
	DB.Where("id LIKE ? AND user_id = ?", "conn_"+id+"%", session.UserID).Delete(&UploadedFile{})

	// Recalculate storage after deletion
	var totalSize int64
	DB.Model(&UploadedFile{}).Where("user_id = ?", session.UserID).Select("COALESCE(SUM(size), 0)").Scan(&totalSize)
	var connFiles []Connection
	DB.Where("user_id = ?", session.UserID).Find(&connFiles)
	var connSizeMB float64
	for _, c := range connFiles {
		if c.CachedTables != "" && c.CachedTables != "null" && c.CachedTables != "[]" {
			var cached struct{ TableDetails []struct{ Rows int `json:"rows"`; Columns int `json:"columns"` } `json:"table_details"` }
			if json.Unmarshal([]byte(c.CachedTables), &cached) == nil {
				for _, t := range cached.TableDetails {
					cols := t.Columns; if cols < 10 { cols = 10 }
					connSizeMB += float64(t.Rows * cols * 20) / (1024 * 1024)
				}
			}
		}
	}
	newStorageMB := float64(totalSize)/(1024*1024) + connSizeMB
	DB.Model(&UserQuota{}).Where("user_id = ?", session.UserID).Update("storage_used_mb", newStorageMB)

	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

func TestConnectionHandler(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Name     string `json:"name"`
		Type     string `json:"type"`
		SubType  string `json:"sub_type"`
		Host     string `json:"host"`
		Port     int    `json:"port"`
		Database string `json:"database"`
		Username string `json:"username"`
		Password string `json:"password"`
		APIKey   string `json:"api_key"`
		Endpoint string `json:"endpoint"`
		Bucket   string `json:"bucket"`
		Region   string `json:"region"`
		SSL      bool   `json:"ssl"`
	}

	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var success bool
	var message string

	switch input.SubType {
	case "postgresql", "supabase":
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			input.Host, input.Port, input.Username, input.Password, input.Database, func() string {
				if input.SubType == "supabase" || input.SSL {
					return "require"
				}
				return "disable"
			}())
		testDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			success = false
			message = "Connection failed: " + err.Error()
		} else {
			sqlDB, _ := testDB.DB()
			if err := sqlDB.Ping(); err != nil {
				success = false
				message = "Ping failed: " + err.Error()
			} else {
				success = true
				message = "Connection successful"
			}
			sqlDB.Close()
		}

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			input.Username, input.Password, input.Host, input.Port, input.Database)
		testDB, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
		if err != nil {
			success = false
			message = "Connection failed: " + err.Error()
		} else {
			sqlDB, _ := testDB.DB()
			if err := sqlDB.Ping(); err != nil {
				success = false
				message = "Ping failed: " + err.Error()
			} else {
				success = true
				message = "Connection successful"
			}
			sqlDB.Close()
		}

	case "mongodb":
		var mongoURI string
		if input.Endpoint != "" {
			mongoURI = input.Endpoint
		} else if input.Host != "" && input.Database != "" {
			mongoURI = fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", input.Username, input.Password, input.Host, input.Port, input.Database)
			if input.Username == "" {
				mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", input.Host, input.Port, input.Database)
			}
		}
		if mongoURI != "" {
			clientOptions := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
			client, err := mongo.Connect(context.Background(), clientOptions)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				err = client.Ping(context.Background(), nil)
				if err != nil {
					success = false
					message = "Ping failed: " + err.Error()
				} else {
					success = true
					message = "MongoDB connection successful"
				}
				client.Disconnect(context.Background())
			}
		} else {
			success = false
			message = "Connection string or host+database required"
		}

	case "pinecone":
		if input.APIKey != "" && input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint+"/describe_index_stats", nil)
			req.Header.Set("Api-Key", input.APIKey)
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					success = true
					message = "Pinecone connection successful"
				} else {
					success = false
					message = fmt.Sprintf("Pinecone returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "API key and endpoint required"
		}

	case "weaviate":
		if input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint+"/v1/.well-known/ready", nil)
			if input.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+input.APIKey)
			}
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					success = true
					message = "Weaviate connection successful"
				} else {
					success = false
					message = fmt.Sprintf("Weaviate returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "Endpoint required"
		}

	case "chroma":
		if input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint+"/api/v1/heartbeat", nil)
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					success = true
					message = "Chroma connection successful"
				} else {
					success = false
					message = fmt.Sprintf("Chroma returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "Endpoint required"
		}

	case "lancedb":
		if input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint+"/v1/table", nil)
			if input.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+input.APIKey)
			}
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode < 500 {
					success = true
					message = "LanceDB connection successful"
				} else {
					success = false
					message = fmt.Sprintf("LanceDB returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "Endpoint/path required"
		}

	case "rest_api":
		if input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint, nil)
			if input.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+input.APIKey)
			}
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				success = true
				message = fmt.Sprintf("REST API reachable (status %d)", resp.StatusCode)
			}
		} else {
			success = false
			message = "Endpoint required"
		}

	case "graphql":
		if input.Endpoint != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("POST", input.Endpoint, nil)
			req.Header.Set("Content-Type", "application/json")
			if input.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+input.APIKey)
			}
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				success = true
				message = fmt.Sprintf("GraphQL endpoint reachable (status %d)", resp.StatusCode)
			}
		} else {
			success = false
			message = "Endpoint required"
		}

	case "google_drive":
		success = false
		message = "Google Drive requires OAuth authentication"

	case "aws_s3":
		if input.APIKey != "" && input.Bucket != "" {
			sess, err := awssession.NewSession(&aws.Config{
				Region:      aws.String(input.Region),
				Credentials: credentials.NewStaticCredentials(input.APIKey, input.Password, ""),
			})
			if err != nil {
				success = false
				message = "Session failed: " + err.Error()
			} else {
				s3Client := s3.New(sess)
				_, err := s3Client.HeadBucket(&s3.HeadBucketInput{
					Bucket: aws.String(input.Bucket),
				})
				if err != nil {
					success = false
					message = "Bucket access failed: " + err.Error()
				} else {
					success = true
					message = "S3 connection successful"
				}
			}
		} else {
			success = false
			message = "Access key and bucket required"
		}

	case "gcs":
		if input.APIKey != "" && input.Bucket != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			url := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s", input.Bucket)
			req, _ := http.NewRequest("GET", url, nil)
			req.Header.Set("Authorization", "Bearer "+input.APIKey)
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					success = true
					message = "GCS connection successful"
				} else {
					success = false
					message = fmt.Sprintf("GCS returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "API key and bucket required"
		}

	case "databricks":
		if (input.Host != "" || input.Endpoint != "") && input.APIKey != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			workspaceURL := input.Host
			if workspaceURL == "" {
				workspaceURL = input.Endpoint
			}
			req, _ := http.NewRequest("GET", workspaceURL+"/api/2.0/clusters/list", nil)
			req.Header.Set("Authorization", "Bearer "+input.APIKey)
			resp, err := client.Do(req)
			if err != nil {
				success = false
				message = "Connection failed: " + err.Error()
			} else {
				resp.Body.Close()
				if resp.StatusCode == 200 {
					success = true
					message = "Databricks connection successful"
				} else {
					success = false
					message = fmt.Sprintf("Databricks returned status %d", resp.StatusCode)
				}
			}
		} else {
			success = false
			message = "Endpoint and token required"
		}

	case "snowflake":
		if input.Host != "" && input.Username != "" && input.Database != "" {
			cfg := &sf.Config{
				Account:   input.Host,
				User:      input.Username,
				Password:  input.Password,
				Database:  input.Database,
				Warehouse: input.Bucket,
			}
			dsn, err := sf.DSN(cfg)
			if err != nil {
				success = false
				message = "Invalid config: " + err.Error()
			} else {
				db, err := sql.Open("snowflake", dsn)
				if err != nil {
					success = false
					message = "Connection failed: " + err.Error()
				} else {
					defer db.Close()
					err = db.Ping()
					if err != nil {
						success = false
						message = "Ping failed: " + err.Error()
					} else {
						success = true
						message = "Snowflake connection successful"
					}
				}
			}
		} else {
			success = false
			message = "Account, username and database required"
		}

	default:
		if input.Host != "" || input.Endpoint != "" {
			success = true
			message = "Connection parameters validated"
		} else {
			success = false
			message = "Host or endpoint required"
		}
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": success,
		"message": message,
	})
}

// UpdateConnectionHandler - Update connection settings
func UpdateConnectionHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, err := GetSession(cookie.Value)
	if err != nil || session == nil {
		http.Error(w, "Invalid session", http.StatusUnauthorized)
		return
	}
	var req struct {
		ID             string   `json:"id"`
		SelectedTables []string `json:"selected_tables"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.ID == "" {
		http.Error(w, "Missing id", http.StatusBadRequest)
		return
	}
	var conn Connection
	if err := DB.Where("id = ? AND user_id = ?", req.ID, session.UserID).First(&conn).Error; err != nil {
		http.Error(w, "Connection not found", http.StatusNotFound)
		return
	}
	if len(req.SelectedTables) > 0 {
		// Filter out empty strings from selected tables
var cleanTables []string
for _, t := range req.SelectedTables {
if strings.TrimSpace(t) != "" { cleanTables = append(cleanTables, t) }
}
tablesJSON, _ := json.Marshal(cleanTables)
		// Build cache from old cache filtered by selected tables
		var oldCache struct {
			Tables       []string `json:"tables"`
			TableDetails []struct {
				Name    string `json:"name"`
				Rows    int64  `json:"rows"`
				Columns int    `json:"columns"`
			} `json:"table_details"`
		}
		if conn.CachedTables != "" {
			json.Unmarshal([]byte(conn.CachedTables), &oldCache)
		}
		selectedMap := make(map[string]bool)
		for _, s := range req.SelectedTables {
			selectedMap[s] = true
		}
		var filteredTables []string
		var filteredInfos []map[string]interface{}
		for _, t := range oldCache.Tables {
			if selectedMap[t] {
				filteredTables = append(filteredTables, t)
			}
		}
		for _, ti := range oldCache.TableDetails {
			if selectedMap[ti.Name] {
				filteredInfos = append(filteredInfos, map[string]interface{}{"name": ti.Name, "rows": ti.Rows, "columns": ti.Columns})
			}
		}
// Databricks: fetch row counts for selected tables
if conn.SubType == "databricks" && conn.Host != "" && conn.APIKey != "" {
httpCl := &http.Client{Timeout: 30 * time.Second}
wURL := "https://" + strings.TrimPrefix(strings.TrimPrefix(conn.Host, "https://"), "http://")
dbCat := conn.Database
if dbCat == "" { dbCat = "main" }
wID := conn.Endpoint
if strings.Contains(wID, "/") { p := strings.Split(wID, "/"); wID = p[len(p)-1] }
filteredInfos = nil
for _, tbl := range req.SelectedTables {
cQ := fmt.Sprintf("SELECT COUNT(*) as cnt FROM %s.%s", dbCat, tbl)
b, _ := json.Marshal(map[string]interface{}{"statement": cQ, "warehouse_id": wID, "wait_timeout": "30s"})
rq, _ := http.NewRequest("POST", wURL+"/api/2.0/sql/statements", bytes.NewReader(b))
rq.Header.Set("Authorization", "Bearer "+conn.APIKey)
rq.Header.Set("Content-Type", "application/json")
rs, er := httpCl.Do(rq)
rw := int64(0)
cl := 0
if er == nil && rs.StatusCode == 200 {
var cr struct { Result struct { DataArray [][]string `json:"data_array"` } `json:"result"` }
json.NewDecoder(rs.Body).Decode(&cr)
rs.Body.Close()
if len(cr.Result.DataArray) > 0 && len(cr.Result.DataArray[0]) > 0 {
if v, e := strconv.ParseInt(cr.Result.DataArray[0][0], 10, 64); e == nil { rw = v }
}
} else if er == nil { rs.Body.Close() }
for _, ti := range oldCache.TableDetails { if ti.Name == tbl { cl = ti.Columns } }
filteredInfos = append(filteredInfos, map[string]interface{}{"name": tbl, "rows": rw, "columns": cl})
log.Printf("📡 Databricks row count: %s = %d rows, %d cols", tbl, rw, cl)
}
filteredTables = req.SelectedTables
}

if (conn.SubType == "postgresql" || conn.SubType == "supabase") && conn.Host != "" {
connHost := conn.Host
sslmode := "disable"
if conn.SubType == "supabase" {
if ips, err := net.LookupIP(conn.Host); err == nil {
for _, ip := range ips { if ip.To4() != nil { connHost = ip.String(); break } }
}
sslmode = "require"
}
if conn.SSL { sslmode = "require" }
dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s", conn.Username, conn.Password, connHost, conn.Port, conn.Database, sslmode)
tempDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
if err == nil {
sqlDB2, _ := tempDB.DB()
filteredInfos = nil
for _, tbl := range req.SelectedTables {
var rw int64
var cl int
sqlDB2.QueryRow("SELECT count(*) FROM \"" + tbl + "\"").Scan(&rw)
sqlDB2.QueryRow("SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND table_name=$1", tbl).Scan(&cl)
filteredInfos = append(filteredInfos, map[string]interface{}{"name": tbl, "rows": rw, "columns": cl})
log.Printf("📡 PostgreSQL row count: %s = %d rows, %d cols", tbl, rw, cl)
}
filteredTables = req.SelectedTables
sqlDB2.Close()
}
}
if conn.SubType == "mongodb" {
var mongoURI string
if conn.Endpoint != "" { mongoURI = conn.Endpoint } else {
mongoURI = fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
if conn.Username == "" { mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database) }
}
clientOpts := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
mclient, merr := mongo.Connect(context.Background(), clientOpts)
if merr == nil {
filteredInfos = nil
dbName := conn.Database
for _, collName := range req.SelectedTables {
cnt, _ := mclient.Database(dbName).Collection(collName).CountDocuments(context.Background(), map[string]interface{}{})
filteredInfos = append(filteredInfos, map[string]interface{}{"name": collName, "rows": cnt, "columns": 0})
log.Printf("📡 MongoDB row count: %s = %d rows", collName, cnt)
}
filteredTables = req.SelectedTables
mclient.Disconnect(context.Background())
}
}
if conn.SubType == "snowflake" && conn.Host != "" {
sfDSN := fmt.Sprintf("%s:%s@%s/%s", conn.Username, conn.Password, conn.Host, conn.Database)
sfDB, err := sql.Open("snowflake", sfDSN)
if err == nil {
filteredInfos = nil
for _, tbl := range req.SelectedTables {
var rw int64
sfDB.QueryRow(fmt.Sprintf("SELECT count(*) FROM %s", tbl)).Scan(&rw)
filteredInfos = append(filteredInfos, map[string]interface{}{"name": tbl, "rows": rw, "columns": 0})
log.Printf("📡 Snowflake row count: %s = %d rows", tbl, rw)
}
filteredTables = req.SelectedTables
sfDB.Close()
}
}
if conn.SubType == "mysql" && conn.Host != "" {
mysqlDSN := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
mysqlDB, err := sql.Open("mysql", mysqlDSN)
if err == nil {
filteredInfos = nil
for _, tbl := range req.SelectedTables {
var rw int64
var cl int
mysqlDB.QueryRow(fmt.Sprintf("SELECT count(*) FROM `%s`", tbl)).Scan(&rw)
mysqlDB.QueryRow("SELECT count(*) FROM information_schema.columns WHERE table_schema=? AND table_name=?", conn.Database, tbl).Scan(&cl)
filteredInfos = append(filteredInfos, map[string]interface{}{"name": tbl, "rows": rw, "columns": cl})
log.Printf("📡 MySQL row count: %s = %d rows, %d cols", tbl, rw, cl)
}
filteredTables = req.SelectedTables
mysqlDB.Close()
}
}
		newCache := map[string]interface{}{"tables": filteredTables}
		if len(filteredInfos) > 0 {
			newCache["table_details"] = filteredInfos
		}
		newCacheBytes, _ := json.Marshal(newCache)
		now := time.Now()
		DB.Model(&conn).Updates(map[string]interface{}{
			"selected_tables": string(tablesJSON),
			"cached_tables":   string(newCacheBytes),
			"cached_at":       now,
		})

	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"success": true})
}

// List tables from a connection
func ListTablesHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	_, err = GetSession(cookie.Value)
	if err != nil {
		http.Error(w, "Invalid session", http.StatusUnauthorized)
		return
	}

session2, _ := GetSession(cookie.Value)
if session2 != nil {
if ok, reason := CheckStorage(session2.UserID, 5); !ok {
http.Error(w, reason, http.StatusForbidden)
return
}
}


	connID := r.URL.Query().Get("connection_id")
	if connID == "" {
		http.Error(w, "Missing connection_id", http.StatusBadRequest)
		return
	}

	var conn Connection
	if err := DB.Where("id = ?", connID).First(&conn).Error; err != nil {
		http.Error(w, "Connection not found", http.StatusNotFound)
		return
	}
	type TableInfo struct {
		Name    string `json:"name"`
		Rows    int64  `json:"rows"`
		Columns int    `json:"columns"`
	}

	// Check cache first (5 min TTL)
forceRefresh := r.URL.Query().Get("refresh") == "true"
isNullCache := conn.CachedTables == "" || conn.CachedTables == `{"tables":null}` || conn.CachedTables == `{"tables":null,"table_details":null}`
if !isNullCache && conn.CachedAt != nil && !forceRefresh {
if conn.SubType == "rest_api" || conn.SubType == "graphql" {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(conn.CachedTables))
		return
	}
if time.Since(*conn.CachedAt) < 24*time.Hour {
w.Header().Set("Content-Type", "application/json")
w.Write([]byte(conn.CachedTables))
return
}
}

	var tableInfos []TableInfo
	var tables []string

	log.Printf("🔍 Tables request: conn=%s type=%s sub_type=%s endpoint=%s host=%s", conn.ID, conn.Type, conn.SubType, conn.Endpoint, conn.Host)

	switch conn.SubType {
	case "postgresql", "supabase":
		connHost := conn.Host
		if conn.SubType == "supabase" {
			if ips, err := net.LookupIP(conn.Host); err == nil {
				for _, ip := range ips {
					if ip.To4() != nil {
						connHost = ip.String()
						break
					}
				}
			}
		}
		sslmode := "disable"
		if conn.SubType == "supabase" || conn.SSL {
			sslmode = "require"
		}
		dsn := fmt.Sprintf("postgresql://%s:%s@%s:%d/%s?sslmode=%s",
			conn.Username, conn.Password, connHost, conn.Port, conn.Database, sslmode)
		tempDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ := tempDB.DB()
		defer sqlDB.Close()

		rows, err := sqlDB.Query(`SELECT t.table_name,
			(SELECT count(*) FROM information_schema.columns c WHERE c.table_schema = 'public' AND c.table_name = t.table_name) as col_count
			FROM information_schema.tables t WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'`)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		for rows.Next() {
			var name string
			var colCount int
			rows.Scan(&name, &colCount)
			tables = append(tables, name)
			// Get row count
			var rowCount int64
			sqlDB.QueryRow("SELECT count(*) FROM \"" + name + "\"").Scan(&rowCount)
			tableInfos = append(tableInfos, TableInfo{Name: name, Rows: rowCount, Columns: colCount})
		}

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		tempDB, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ := tempDB.DB()
		defer sqlDB.Close()

		rows, err := sqlDB.Query("SHOW TABLES")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		for rows.Next() {
			var name string
			rows.Scan(&name)
			tables = append(tables, name)
			var rowCount int64
			sqlDB.QueryRow("SELECT COUNT(*) FROM `" + name + "`").Scan(&rowCount)
			var colCount int
			sqlDB.QueryRow("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = ? AND table_name = ?", conn.Database, name).Scan(&colCount)
			tableInfos = append(tableInfos, TableInfo{Name: name, Rows: rowCount, Columns: colCount})
		}

	case "snowflake":
		cfg := &sf.Config{
			Account:   conn.Host,
			User:      conn.Username,
			Password:  conn.Password,
			Database:  conn.Database,
			Warehouse: conn.Bucket,
		}
		sfDsn, _ := sf.DSN(cfg)
		sfDB, err := sql.Open("snowflake", sfDsn)
log.Printf("🔍 Snowflake open: err=%v account=%s db=%s", err, conn.Host, conn.Database)
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer sfDB.Close()
		// Try SHOW TABLES with schema
sfQuery := "SHOW TABLES IN DATABASE " + conn.Database
sfRows, err := sfDB.Query(sfQuery)
log.Printf("🔍 Snowflake query: %s err=%v", sfQuery, err)
if err != nil {
sfQuery = "SHOW TABLES"
sfRows, err = sfDB.Query(sfQuery)
log.Printf("🔍 Snowflake fallback query: %s err=%v", sfQuery, err)
}
// Also try listing schemas
schRows, schErr := sfDB.Query("SHOW SCHEMAS IN DATABASE " + conn.Database)
if schErr == nil {
for schRows.Next() {
var schVals [11]string
schRows.Scan(&schVals[0],&schVals[1],&schVals[2],&schVals[3],&schVals[4],&schVals[5],&schVals[6],&schVals[7],&schVals[8],&schVals[9],&schVals[10])
log.Printf("🔍 Snowflake schema: %s", schVals[1])
}
schRows.Close()
}
log.Printf("🔍 Snowflake SHOW TABLES: err=%v", err)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer sfRows.Close()
		for sfRows.Next() {
			sfCols, _ := sfRows.Columns()
			vals := make([]interface{}, len(sfCols))
			for i := range vals { vals[i] = new(sql.NullString) }
			scanErr := sfRows.Scan(vals...)
			if scanErr != nil { log.Printf("❌ Snowflake scan error: %v", scanErr); continue }
			name := ""
			var sfRowsFromMeta int64
			for i, col := range sfCols {
				v := vals[i].(*sql.NullString)
				if !v.Valid { continue }
				switch strings.ToLower(col) {
				case "name": name = v.String
				case "rows":
					if rc, e := strconv.ParseInt(v.String, 10, 64); e == nil { sfRowsFromMeta = rc }
				}
			}
			log.Printf("🔍 Snowflake table: %s rows=%d", name, sfRowsFromMeta)
			if name == "" { continue }
			tables = append(tables, name)
			rowCount := sfRowsFromMeta
			if rowCount == 0 {
				sfDB.QueryRow("SELECT COUNT(*) FROM \"" + name + "\"").Scan(&rowCount)
			}
			var colCount int
			sfDB.QueryRow("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = ?", name).Scan(&colCount)
			tableInfos = append(tableInfos, TableInfo{Name: name, Rows: rowCount, Columns: colCount})
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
		clientOptions := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
		client, err := mongo.Connect(context.Background(), clientOptions)
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer client.Disconnect(context.Background())
		collections, err := client.Database(conn.Database).ListCollectionNames(context.Background(), map[string]interface{}{})
		if err != nil {
			http.Error(w, "Failed to list collections: "+err.Error(), http.StatusInternalServerError)
			return
		}
		tables = collections
		for _, collName := range collections {
			coll := client.Database(conn.Database).Collection(collName)
			count, _ := coll.CountDocuments(context.Background(), map[string]interface{}{})
			tableInfos = append(tableInfos, TableInfo{Name: collName, Rows: count, Columns: 0})
		}

	case "databricks":
		if conn.Host != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			workspaceURL := "https://" + strings.TrimPrefix(strings.TrimPrefix(conn.Host, "https://"), "http://")
			catalog := conn.Database
			if catalog == "" {
				catalog = "main"
			}
			// List schemas first
			schReq, _ := http.NewRequest("GET", workspaceURL+"/api/2.1/unity-catalog/schemas?catalog_name="+catalog, nil)
			schReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
			schResp, serr := httpClient.Do(schReq)
			var schemaNames []string
			if serr == nil && schResp.StatusCode == 200 {
				var schResult struct {
					Schemas []struct {
						Name string `json:"name"`
					} `json:"schemas"`
				}
				json.NewDecoder(schResp.Body).Decode(&schResult)
				schResp.Body.Close()
				for _, s := range schResult.Schemas {
					if s.Name != "information_schema" {
						schemaNames = append(schemaNames, s.Name)
					}
				}
			} else if serr == nil {
				schResp.Body.Close()
			}
			if len(schemaNames) == 0 {
				schemaNames = []string{"default"}
			}
			for _, schema := range schemaNames {
				tReq, _ := http.NewRequest("GET", workspaceURL+"/api/2.1/unity-catalog/tables?catalog_name="+catalog+"&schema_name="+schema, nil)
				tReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
				tResp, terr := httpClient.Do(tReq)
				if terr == nil && tResp.StatusCode == 200 {
					var tResult struct {
						Tables []struct {
							Name    string `json:"name"`
							Columns []struct {
								Name string `json:"name"`
							} `json:"columns"`
						} `json:"tables"`
					}
					json.NewDecoder(tResp.Body).Decode(&tResult)
					tResp.Body.Close()
					for _, t := range tResult.Tables {
						tableFull := schema + "." + t.Name
						tables = append(tables, tableFull)
						// Try to get row count via SQL API
dbRowCount := int64(0)
wID := conn.Endpoint
if strings.Contains(wID, "/") { pp := strings.Split(wID, "/"); wID = pp[len(pp)-1] }
if wID != "" {
countQ := fmt.Sprintf("SELECT COUNT(*) as cnt FROM %s.%s", catalog, tableFull)
cBody, _ := json.Marshal(map[string]interface{}{"statement": countQ, "warehouse_id": wID, "wait_timeout": "30s"})
cReq, _ := http.NewRequest("POST", workspaceURL+"/api/2.0/sql/statements", bytes.NewReader(cBody))
cReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
cReq.Header.Set("Content-Type", "application/json")
cResp, cErr := httpClient.Do(cReq)
if cErr == nil && cResp.StatusCode == 200 {
var cResult struct { Result struct { DataArray [][]string `json:"data_array"` } `json:"result"` }
json.NewDecoder(cResp.Body).Decode(&cResult)
cResp.Body.Close()
if len(cResult.Result.DataArray) > 0 && len(cResult.Result.DataArray[0]) > 0 {
if v, e := strconv.ParseInt(cResult.Result.DataArray[0][0], 10, 64); e == nil { dbRowCount = v }
}
} else if cErr == nil { cResp.Body.Close() }
}
tableInfos = append(tableInfos, TableInfo{Name: tableFull, Rows: dbRowCount, Columns: len(t.Columns)})
log.Printf("📡 Databricks table refresh: %s = %d rows, %d cols", tableFull, dbRowCount, len(t.Columns))
					}
				} else if terr == nil {
					tResp.Body.Close()
				}
			}
		}

	case "pinecone":
		if conn.Endpoint != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("POST", conn.Endpoint+"/describe_index_stats", strings.NewReader("{}"))
			req.Header.Set("Api-Key", conn.APIKey)
			req.Header.Set("Content-Type", "application/json")
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var stats struct {
					TotalVectorCount int64 `json:"totalVectorCount"`
					Dimension        int   `json:"dimension"`
					Namespaces       map[string]struct {
						VectorCount int64 `json:"vectorCount"`
					} `json:"namespaces"`
				}
				json.NewDecoder(resp.Body).Decode(&stats)
				if len(stats.Namespaces) > 0 {
					for ns, info := range stats.Namespaces {
						name := ns
						if name == "" {
							name = "default"
						}
						tables = append(tables, name)
						tableInfos = append(tableInfos, TableInfo{Name: name, Rows: info.VectorCount, Columns: stats.Dimension})
					}
				} else {
					tables = append(tables, "vectors")
					tableInfos = append(tableInfos, TableInfo{Name: "vectors", Rows: stats.TotalVectorCount, Columns: stats.Dimension})
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "weaviate":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/v1/schema", nil)
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var schema struct {
					Classes []struct {
						Class      string `json:"class"`
						Properties []struct {
							Name string `json:"name"`
						} `json:"properties"`
					} `json:"classes"`
				}
				json.NewDecoder(resp.Body).Decode(&schema)
				for _, cls := range schema.Classes {
					tables = append(tables, cls.Class)
					tableInfos = append(tableInfos, TableInfo{Name: cls.Class, Rows: 0, Columns: len(cls.Properties)})
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "chroma":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			// Build Chroma URL with tenant/database for Cloud (v2 API)
chromaBase := strings.TrimRight(conn.Endpoint, "/")
chromaTenant := "default_tenant"
if conn.Database != "" { chromaTenant = conn.Database }
chromaDB := "default_database"
if conn.Bucket != "" { chromaDB = conn.Bucket }
collectionsURL := chromaBase + "/api/v2/tenants/" + chromaTenant + "/databases/" + chromaDB + "/collections"
log.Printf("🔍 Chroma collections URL: %s", collectionsURL)
req, _ := http.NewRequest("GET", collectionsURL, nil)
			if conn.APIKey != "" {
				req.Header.Set("X-Chroma-Token", conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var collections []struct {
					ID   string `json:"id"`
					Name string `json:"name"`
				}
				json.NewDecoder(resp.Body).Decode(&collections)
				for _, coll := range collections {
					tables = append(tables, coll.Name)
					// Get count per collection
					countReq, _ := http.NewRequest("GET", chromaBase+"/api/v2/tenants/"+chromaTenant+"/databases/"+chromaDB+"/collections/"+coll.ID+"/count", nil)
					if conn.APIKey != "" {
						countReq.Header.Set("X-Chroma-Token", conn.APIKey)
					}
					countResp, cerr := httpClient.Do(countReq)
					var count int64
					if cerr == nil && countResp.StatusCode == 200 {
						json.NewDecoder(countResp.Body).Decode(&count)
						countResp.Body.Close()
					} else if cerr == nil {
						countResp.Body.Close()
					}
					tableInfos = append(tableInfos, TableInfo{Name: coll.Name, Rows: count, Columns: 0})
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "lancedb":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/v1/table/", nil)
			if conn.APIKey != "" {
				req.Header.Set("x-api-key", conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var tableNames []string
				json.NewDecoder(resp.Body).Decode(&tableNames)
				for _, t := range tableNames {
					tables = append(tables, t)
					tableInfos = append(tableInfos, TableInfo{Name: t, Rows: 0, Columns: 0})
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "rest_api":
		if conn.Endpoint != "" {
log.Printf("🔍 REST API ListTables: endpoint=%s", conn.Endpoint)
			httpClient := &http.Client{Timeout: 30 * time.Second}
			req, _ := http.NewRequest("GET", conn.Endpoint, nil)
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
if err == nil {
rlLimit := resp.Header.Get("X-RateLimit-Limit")
if rlLimit == "" { rlLimit = resp.Header.Get("X-Rate-Limit-Limit") }
if rlLimit == "" { rlLimit = resp.Header.Get("RateLimit-Limit") }
rlRemaining := resp.Header.Get("X-RateLimit-Remaining")
if rlRemaining == "" { rlRemaining = resp.Header.Get("X-Rate-Limit-Remaining") }
if rlRemaining == "" { rlRemaining = resp.Header.Get("RateLimit-Remaining") }
rlTotal := resp.Header.Get("X-Total-Count")
if rlTotal == "" { rlTotal = resp.Header.Get("X-Total") }
if rlTotal != "" && rlLimit == "" {
DB.Model(&conn).Update("rate_limit", rlTotal + " total records")
log.Printf("📊 REST API total: %s", rlTotal)
}
if rlLimit != "" {
rlStr := rlRemaining + "/" + rlLimit
if rlRemaining == "" { rlStr = rlLimit + " req limit" }
DB.Model(&conn).Update("rate_limit", rlStr)
log.Printf("📊 REST API rate limit: %s", rlStr)
}
if resp.StatusCode == 429 {
DB.Model(&conn).Update("rate_limit", "Rate limited")
log.Printf("📊 REST API rate limited!")
}
}
if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
				var jsonArray []interface{}
				if json.Unmarshal(bodyBytes, &jsonArray) == nil && len(jsonArray) > 0 {
					tables = append(tables, "api_data")
					cols := 0
					if first, ok := jsonArray[0].(map[string]interface{}); ok {
						cols = len(first)
					}
					tableInfos = append(tableInfos, TableInfo{Name: "api_data", Rows: int64(len(jsonArray)), Columns: cols})
rowCount := len(jsonArray)
if rowCount == 1000 || rowCount == 500 || rowCount == 100 || rowCount == 10000 || rowCount == 5000 || rowCount == 2000 || rowCount == 50 || rowCount == 25 || rowCount == 200 {
if conn.RateLimit == "" {
DB.Model(&conn).Update("rate_limit", fmt.Sprintf("%d rows/request (API default limit)", rowCount))
log.Printf("📊 REST API default limit detected: %d rows", rowCount)
}
}
				} else {
					var jsonObj map[string]interface{}
					if json.Unmarshal(bodyBytes, &jsonObj) == nil {
						for key, val := range jsonObj {
							if arr, ok := val.([]interface{}); ok {
								tables = append(tables, key)
								cols := 0
								if len(arr) > 0 {
									if first, ok := arr[0].(map[string]interface{}); ok {
										cols = len(first)
									}
								}
								tableInfos = append(tableInfos, TableInfo{Name: key, Rows: int64(len(arr)), Columns: cols})
							}
						}
					}
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "graphql":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			typesQuery := `{"query":"{ __schema { types { name kind fields { name type { name kind ofType { name kind ofType { name kind } } } } } } }"}`
			req, _ := http.NewRequest("POST", conn.Endpoint, strings.NewReader(typesQuery))
			req.Header.Set("Content-Type", "application/json")
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				respBytes, _ := io.ReadAll(resp.Body)
				var rawResult map[string]interface{}
				json.Unmarshal(respBytes, &rawResult)
				dataObj, _ := rawResult["data"].(map[string]interface{})
				schemaObj, _ := dataObj["__schema"].(map[string]interface{})
				allTypes, _ := schemaObj["types"].([]interface{})
				// Build type->col count and scalar fields map
				typeColCount := make(map[string]int)
				typeScalarFields := make(map[string][]string)
				for _, typ := range allTypes {
					tObj, _ := typ.(map[string]interface{})
					tName, _ := tObj["name"].(string)
					tKind, _ := tObj["kind"].(string)
					if tKind != "OBJECT" || strings.HasPrefix(tName, "__") {
						continue
					}
					fields, _ := tObj["fields"].([]interface{})
					var scalars []string
					for _, fld := range fields {
						fObj, _ := fld.(map[string]interface{})
						fName, _ := fObj["name"].(string)
						fType, _ := fObj["type"].(map[string]interface{})
						fKind, _ := fType["kind"].(string)
						ftName, _ := fType["name"].(string)
						if fKind == "NON_NULL" {
							if ot, ok := fType["ofType"].(map[string]interface{}); ok {
								ftName, _ = ot["name"].(string)
							}
						}
						if fKind == "SCALAR" || ftName == "String" || ftName == "Int" || ftName == "Float" || ftName == "Boolean" || ftName == "ID" {
							scalars = append(scalars, fName)
						}
					}
					typeColCount[tName] = len(fields)
					if len(scalars) > 0 {
						typeScalarFields[tName] = scalars
					}
				}
				// Helper: check if type chain has LIST and resolve return type name
				isListAndResolve := func(typeObj map[string]interface{}) (bool, string) {
					isList := false
					retName := ""
					cur := typeObj
					for cur != nil {
						k, _ := cur["kind"].(string)
						n, _ := cur["name"].(string)
						if k == "LIST" {
							isList = true
						}
						if (k == "OBJECT" || k == "INTERFACE") && n != "" {
							retName = n
						}
						if ot, ok := cur["ofType"].(map[string]interface{}); ok {
							cur = ot
						} else {
							break
						}
					}
					return isList, retName
				}
				// Find Query type - only show list-returning fields as tables
				for _, typ := range allTypes {
					tObj, _ := typ.(map[string]interface{})
					tName, _ := tObj["name"].(string)
					if tName != "Query" {
						continue
					}
					fields, _ := tObj["fields"].([]interface{})
					for _, fld := range fields {
						fObj, _ := fld.(map[string]interface{})
						fType, _ := fObj["type"].(map[string]interface{})
						isList, retType := isListAndResolve(fType)
						if !isList {
							continue
						}
						fName, _ := fObj["name"].(string)
						if retType == "" {
							singular := strings.TrimSuffix(fName, "ies")
							if singular != fName {
								singular += "y"
							} else {
								singular = strings.TrimSuffix(fName, "s")
							}
							cap := strings.ToUpper(singular[:1]) + singular[1:]
							if _, ok := typeColCount[cap]; ok {
								retType = cap
							}
						}
						if retType == "" {
							continue
						}
						cols := typeColCount[retType]
						tables = append(tables, retType)
						tableInfos = append(tableInfos, TableInfo{Name: retType, Rows: 0, Columns: cols})
					}
					break
				}
				// Fetch row counts using list queries
				for _, typ := range allTypes {
					tObj, _ := typ.(map[string]interface{})
					tName, _ := tObj["name"].(string)
					if tName != "Query" {
						continue
					}
					fields, _ := tObj["fields"].([]interface{})
					for _, fld := range fields {
						fObj, _ := fld.(map[string]interface{})
						fType, _ := fObj["type"].(map[string]interface{})
						isList, _ := isListAndResolve(fType)
						if !isList {
							continue
						}
						queryName, _ := fObj["name"].(string)
						singular := strings.TrimSuffix(queryName, "ies")
						if singular != queryName {
							singular += "y"
						} else {
							singular = strings.TrimSuffix(queryName, "s")
						}
						cap := strings.ToUpper(singular[:1]) + singular[1:]
						matchIdx := -1
						for ti, tbl := range tableInfos {
							if tbl.Name == cap {
								matchIdx = ti
								break
							}
						}
						if matchIdx < 0 {
							continue
						}
						scalars := typeScalarFields[tableInfos[matchIdx].Name]
						if len(scalars) == 0 {
							continue
						}
						gqlQ := fmt.Sprintf(`{"query":"{ %s { %s } }"}`, queryName, scalars[0])
						cReq, _ := http.NewRequest("POST", conn.Endpoint, strings.NewReader(gqlQ))
						cReq.Header.Set("Content-Type", "application/json")
						if conn.APIKey != "" {
							cReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
						}
						cResp, cerr := httpClient.Do(cReq)
						if cerr == nil && cResp.StatusCode == 200 {
							var cResult map[string]interface{}
							json.NewDecoder(cResp.Body).Decode(&cResult)
							cResp.Body.Close()
							if d, ok := cResult["data"].(map[string]interface{}); ok {
								if arr, ok := d[queryName].([]interface{}); ok {
									tableInfos[matchIdx].Rows = int64(len(arr))
								}
							}
						} else if cerr == nil {
							cResp.Body.Close()
						}
					}
					break
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "google_drive", "google-drive":
		if conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("GET", "https://www.googleapis.com/drive/v3/files?q=mimeType%3D'application/vnd.google-apps.spreadsheet'+or+mimeType%3D'text/csv'&fields=files(id,name,mimeType)&pageSize=100", nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var result struct {
					Files []struct {
						ID       string `json:"id"`
						Name     string `json:"name"`
						MimeType string `json:"mimeType"`
					} `json:"files"`
				}
				json.NewDecoder(resp.Body).Decode(&result)
				for _, f := range result.Files {
					tables = append(tables, f.Name)
					tableInfos = append(tableInfos, TableInfo{Name: f.Name, Rows: 0, Columns: 0})
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}

	case "aws_s3", "aws-s3":
		if conn.Bucket != "" && conn.APIKey != "" {
			// AWS S3 ListObjectsV2 via REST API
			region := conn.Region
			if region == "" {
				region = "us-east-1"
			}
			s3URL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/?list-type=2&max-keys=100", conn.Bucket, region)
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("GET", s3URL, nil)
			// For public buckets or pre-signed - try without auth first
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				bodyBytes, _ := io.ReadAll(resp.Body)
				bodyStr := string(bodyBytes)
				// Parse XML keys
				keyStart := 0
				for {
					idx := strings.Index(bodyStr[keyStart:], "<Key>")
					if idx == -1 {
						break
					}
					keyStart += idx + 5
					endIdx := strings.Index(bodyStr[keyStart:], "</Key>")
					if endIdx == -1 {
						break
					}
					objName := bodyStr[keyStart : keyStart+endIdx]
					keyStart += endIdx + 6
					if strings.HasSuffix(objName, ".csv") || strings.HasSuffix(objName, ".json") || strings.HasSuffix(objName, ".parquet") || strings.HasSuffix(objName, ".xlsx") {
						tables = append(tables, objName)
						tableInfos = append(tableInfos, TableInfo{Name: objName, Rows: 0, Columns: 0})
					}
				}
			} else if err == nil {
				resp.Body.Close()
				// Fallback - just show bucket name
				tables = append(tables, conn.Bucket)
				tableInfos = append(tableInfos, TableInfo{Name: conn.Bucket, Rows: 0, Columns: 0})
			}
		}

	case "gcs":
		if conn.Bucket != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 15 * time.Second}
			req, _ := http.NewRequest("GET", "https://storage.googleapis.com/storage/v1/b/"+conn.Bucket+"/o?maxResults=100", nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err == nil && resp.StatusCode == 200 {
				defer resp.Body.Close()
				var result struct {
					Items []struct {
						Name string `json:"name"`
						Size string `json:"size"`
					} `json:"items"`
				}
				json.NewDecoder(resp.Body).Decode(&result)
				for _, item := range result.Items {
					if strings.HasSuffix(item.Name, ".csv") || strings.HasSuffix(item.Name, ".json") || strings.HasSuffix(item.Name, ".parquet") {
						tables = append(tables, item.Name)
						tableInfos = append(tableInfos, TableInfo{Name: item.Name, Rows: 0, Columns: 0})
					}
				}
			} else if err == nil {
				resp.Body.Close()
			}
		}
	}

	// Filter by selected_tables if set
	if conn.SelectedTables != "" {
		var selectedList []string
		json.Unmarshal([]byte(conn.SelectedTables), &selectedList)
		// Remove empty strings
		var cleanSelected []string
		for _, s := range selectedList {
			if strings.TrimSpace(s) != "" { cleanSelected = append(cleanSelected, s) }
		}
		selectedList = cleanSelected
		if len(selectedList) > 0 {
			selectedMap := make(map[string]bool)
			for _, s := range selectedList {
				selectedMap[s] = true
			}
			var filteredTables []string
			var filteredInfos []TableInfo
			for _, t := range tables {
				if selectedMap[t] {
					filteredTables = append(filteredTables, t)
				}
			}
			for _, ti := range tableInfos {
				if selectedMap[ti.Name] {
					filteredInfos = append(filteredInfos, ti)
				}
			}
			tables = filteredTables
			tableInfos = filteredInfos
		}
	}
	// Preserve old row counts if new query returned 0
	if conn.CachedTables != "" && len(tableInfos) > 0 {
		var prevCache struct {
			TableDetails []struct {
				Name    string `json:"name"`
				Rows    int64  `json:"rows"`
				Columns int    `json:"columns"`
			} `json:"table_details"`
		}
		json.Unmarshal([]byte(conn.CachedTables), &prevCache)
		prevMap := make(map[string]int64)
		prevColMap := make(map[string]int)
		for _, pt := range prevCache.TableDetails {
			if pt.Rows > 0 { prevMap[pt.Name] = pt.Rows }
			if pt.Columns > 0 { prevColMap[pt.Name] = pt.Columns }
		}
		for idx, ti := range tableInfos {
			if ti.Rows == 0 {
				if oldRows, ok := prevMap[ti.Name]; ok {
					tableInfos[idx] = TableInfo{Name: ti.Name, Rows: oldRows, Columns: ti.Columns}
					log.Printf("🔒 Preserved cached row count for %s: %d rows", ti.Name, oldRows)
				}
			}
			if ti.Columns == 0 {
				if oldCols, ok := prevColMap[ti.Name]; ok {
					tableInfos[idx] = TableInfo{Name: tableInfos[idx].Name, Rows: tableInfos[idx].Rows, Columns: oldCols}
				}
			}
		}
	}
	// Cache results to DB
	responseData := map[string]interface{}{"tables": tables}
	if len(tableInfos) > 0 {
		responseData["table_details"] = tableInfos
	}
	cacheBytes, _ := json.Marshal(responseData)
	now := time.Now()
	DB.Model(&Connection{}).Where("id = ?", conn.ID).Updates(map[string]interface{}{"cached_tables": string(cacheBytes), "cached_at": now})
	json.NewEncoder(w).Encode(responseData)
}

// Export table to CSV
func ExportTableHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, err := GetSession(cookie.Value)
	if err != nil {
		http.Error(w, "Invalid session", http.StatusUnauthorized)
		return
	}

	var input struct {
		ConnectionID string `json:"connection_id"`
		TableName    string `json:"table_name"`
		Limit        int    `json:"limit"`
	}

	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if input.Limit == 0 {
		input.Limit = 10000 // Default limit
	}

	var conn Connection
	if err := DB.Where("id = ?", input.ConnectionID).First(&conn).Error; err != nil {
		http.Error(w, "Connection not found", http.StatusNotFound)
		return
	}

	var rows *sql.Rows
	var sqlDB *sql.DB

	switch conn.SubType {
	case "postgresql", "supabase":
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			conn.Host, conn.Port, conn.Username, conn.Password, conn.Database, func() string {
				if conn.SubType == "supabase" || conn.SSL {
					return "require"
				}
				return "disable"
			}())
		tempDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ = tempDB.DB()
		defer sqlDB.Close()

		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", sanitizeTableName(input.TableName), input.Limit)
		rows, err = sqlDB.Query(query)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

	case "mysql":
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
			conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		tempDB, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ = tempDB.DB()
		defer sqlDB.Close()

		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", sanitizeTableName(input.TableName), input.Limit)
		rows, err = sqlDB.Query(query)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

	case "snowflake":
		sfCfg := &sf.Config{
			Account:   conn.Host,
			User:      conn.Username,
			Password:  conn.Password,
			Database:  conn.Database,
			Warehouse: conn.Bucket,
		}
		sfExpDsn, _ := sf.DSN(sfCfg)
		sfExpDB, err := sql.Open("snowflake", sfExpDsn)
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB = sfExpDB
		defer sqlDB.Close()
		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", sanitizeTableName(input.TableName), input.Limit)
		rows, err = sqlDB.Query(query)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
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
		clientOptions := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
		mongoClient, err := mongo.Connect(context.Background(), clientOptions)
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer mongoClient.Disconnect(context.Background())

		collection := mongoClient.Database(conn.Database).Collection(sanitizeTableName(input.TableName))
		cursor, err := collection.Find(context.Background(), map[string]interface{}{}, options.Find().SetLimit(int64(input.Limit)))
		if err != nil {
			http.Error(w, "Query failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer cursor.Close(context.Background())

		var results []map[string]interface{}
		if err := cursor.All(context.Background(), &results); err != nil {
			http.Error(w, "Failed to decode: "+err.Error(), http.StatusInternalServerError)
			return
		}

		fileID := generateSessionID()[:16]
		filename := fmt.Sprintf("%s_%s.csv", sanitizeFilename(conn.Database), sanitizeTableName(input.TableName))
		filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
		file, _ := os.Create(filepath)
		defer file.Close()
		csvWriter := csv.NewWriter(file)

		if len(results) > 0 {
			var headers []string
			for k := range results[0] {
				headers = append(headers, k)
			}
			csvWriter.Write(headers)
			for _, doc := range results {
				var row []string
				for _, h := range headers {
					row = append(row, fmt.Sprintf("%v", doc[h]))
				}
				csvWriter.Write(row)
			}
		}
		csvWriter.Flush()

		fileInfo, _ := os.Stat(filepath)
		fileSize := fileInfo.Size()
		uploadedFile := UploadedFile{
			ID:        fileID,
			Filename:  filename,
			Path:      filepath,
			Size:      fileSize,
			UserID:    session.UserID,
			CreatedAt: time.Now(),
		}
		DB.Create(&uploadedFile)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"file_id":  fileID,
			"filename": filename,
			"rows":     len(results),
			"columns":  len(results),
		})
		return

	case "databricks":
		if conn.Host != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", sanitizeTableName(input.TableName), input.Limit)
			reqBody, _ := json.Marshal(map[string]interface{}{"statement": query, "warehouse_id": conn.Endpoint})
			workspaceURL := "https://" + strings.TrimPrefix(strings.TrimPrefix(conn.Host, "https://"), "http://")
			req, _ := http.NewRequest("POST", workspaceURL+"/api/2.0/sql/statements", bytes.NewReader(reqBody))
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			req.Header.Set("Content-Type", "application/json")
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "Databricks query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			var result struct {
				Manifest struct {
					Schema struct {
						Columns []struct {
							Name string `json:"name"`
						} `json:"columns"`
					} `json:"schema"`
				} `json:"manifest"`
				Result struct {
					DataArray [][]string `json:"data_array"`
				} `json:"result"`
			}
			json.NewDecoder(resp.Body).Decode(&result)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("databricks_%s.csv", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			file, _ := os.Create(filepath)
			csvWriter := csv.NewWriter(file)
			var headers []string
			for _, c := range result.Manifest.Schema.Columns {
				headers = append(headers, c.Name)
			}
			csvWriter.Write(headers)
			for _, row := range result.Result.DataArray {
				csvWriter.Write(row)
			}
			csvWriter.Flush()
			file.Close()
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": len(result.Result.DataArray), "columns": len(headers), "size": fileInfo.Size()})
			return
		}

	case "pinecone":
		if conn.Endpoint != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			ns := input.TableName
			if ns == "default" {
				ns = ""
			}
			// Using list endpoint instead of query
			// Use list endpoint instead
			listURL := conn.Endpoint + "/vectors/list"
			if ns != "" {
				listURL += "?namespace=" + ns
			}
			listURL += "&limit=" + fmt.Sprintf("%d", input.Limit)
			req, _ := http.NewRequest("GET", listURL, nil)
			req.Header.Set("Api-Key", conn.APIKey)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "Pinecone query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			var result struct {
				Vectors []struct {
					ID       string                 `json:"id"`
					Metadata map[string]interface{} `json:"metadata"`
				} `json:"vectors"`
			}
			json.NewDecoder(resp.Body).Decode(&result)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("pinecone_%s.csv", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			file, _ := os.Create(filepath)
			csvWriter := csv.NewWriter(file)
			headers := []string{"id"}
			if len(result.Vectors) > 0 && result.Vectors[0].Metadata != nil {
				for k := range result.Vectors[0].Metadata {
					headers = append(headers, k)
				}
			}
			csvWriter.Write(headers)
			for _, v := range result.Vectors {
				row := []string{v.ID}
				for _, h := range headers[1:] {
					row = append(row, fmt.Sprintf("%v", v.Metadata[h]))
				}
				csvWriter.Write(row)
			}
			csvWriter.Flush()
			file.Close()
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": len(result.Vectors), "columns": len(headers), "size": fileInfo.Size()})
			return
		}

	case "weaviate":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			gqlQuery := fmt.Sprintf(`{"query":"{ Get { %s(limit: %d) { _additional { id } } } }"}`, input.TableName, input.Limit)
			req, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/v1/graphql", strings.NewReader(gqlQuery))
			req.Header.Set("Content-Type", "application/json")
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "Weaviate query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("weaviate_%s.json", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": 0, "columns": 0, "size": fileInfo.Size()})
			return
		}

	case "chroma":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// Get collection ID first
			req, _ := http.NewRequest("GET", strings.TrimRight(conn.Endpoint, "/")+"/api/v1/collections", nil)
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "Chroma query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			var collections []struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			}
			json.NewDecoder(resp.Body).Decode(&collections)
			var collID string
			for _, c := range collections {
				if c.Name == input.TableName {
					collID = c.ID
					break
				}
			}
			if collID == "" {
				http.Error(w, "Collection not found", http.StatusNotFound)
				return
			}
			// Get documents
			getBody, _ := json.Marshal(map[string]interface{}{"limit": input.Limit, "include": []string{"documents", "metadatas"}})
			getReq, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/api/v1/collections/"+collID+"/get", bytes.NewReader(getBody))
			getReq.Header.Set("Content-Type", "application/json")
			if conn.APIKey != "" {
				getReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			getResp, gerr := httpClient.Do(getReq)
			if gerr != nil {
				http.Error(w, "Chroma get failed: "+gerr.Error(), http.StatusInternalServerError)
				return
			}
			defer getResp.Body.Close()
			var getResult struct {
				IDs       []string                 `json:"ids"`
				Documents []string                 `json:"documents"`
				Metadatas []map[string]interface{} `json:"metadatas"`
			}
			json.NewDecoder(getResp.Body).Decode(&getResult)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("chroma_%s.csv", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			file, _ := os.Create(filepath)
			csvWriter := csv.NewWriter(file)
			headers := []string{"id", "document"}
			if len(getResult.Metadatas) > 0 && getResult.Metadatas[0] != nil {
				for k := range getResult.Metadatas[0] {
					headers = append(headers, k)
				}
			}
			csvWriter.Write(headers)
			for i, id := range getResult.IDs {
				row := []string{id}
				if i < len(getResult.Documents) {
					row = append(row, getResult.Documents[i])
				} else {
					row = append(row, "")
				}
				if i < len(getResult.Metadatas) && getResult.Metadatas[i] != nil {
					for _, h := range headers[2:] {
						row = append(row, fmt.Sprintf("%v", getResult.Metadatas[i][h]))
					}
				}
				csvWriter.Write(row)
			}
			csvWriter.Flush()
			file.Close()
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": len(getResult.IDs), "columns": len(headers), "size": fileInfo.Size()})
			return
		}

	case "lancedb":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			reqBody, _ := json.Marshal(map[string]interface{}{"limit": input.Limit})
			req, _ := http.NewRequest("POST", strings.TrimRight(conn.Endpoint, "/")+"/v1/table/"+input.TableName+"/query", bytes.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			if conn.APIKey != "" {
				req.Header.Set("x-api-key", conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "LanceDB query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("lancedb_%s.json", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": 0, "columns": 0, "size": fileInfo.Size()})
			return
		}

	case "rest_api":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			req, _ := http.NewRequest("GET", conn.Endpoint, nil)
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "API request failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
			var jsonArray []map[string]interface{}
			if json.Unmarshal(bodyBytes, &jsonArray) == nil && len(jsonArray) > 0 {
				fileID := generateSessionID()[:16]
				filename := "api_data.csv"
				filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
				file, _ := os.Create(filepath)
				csvWriter := csv.NewWriter(file)
				var headers []string
				for k := range jsonArray[0] {
					headers = append(headers, k)
				}
				csvWriter.Write(headers)
				for _, obj := range jsonArray {
					var row []string
					for _, h := range headers {
						row = append(row, fmt.Sprintf("%v", obj[h]))
					}
					csvWriter.Write(row)
				}
				csvWriter.Flush()
				file.Close()
				fileInfo, _ := os.Stat(filepath)
				uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
				DB.Create(&uploadedFile)
				json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": len(jsonArray), "columns": len(headers), "size": fileInfo.Size()})
				return
			}
			// Try as object with arrays
			var jsonObj map[string]interface{}
			if json.Unmarshal(bodyBytes, &jsonObj) == nil {
				for key, val := range jsonObj {
					if key == input.TableName {
						if arr, ok := val.([]interface{}); ok && len(arr) > 0 {
							fileID := generateSessionID()[:16]
							filename := fmt.Sprintf("api_%s.csv", key)
							filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
							file, _ := os.Create(filepath)
							csvWriter := csv.NewWriter(file)
							if first, ok := arr[0].(map[string]interface{}); ok {
								var headers []string
								for k := range first {
									headers = append(headers, k)
								}
								csvWriter.Write(headers)
								for _, item := range arr {
									if obj, ok := item.(map[string]interface{}); ok {
										var row []string
										for _, h := range headers {
											row = append(row, fmt.Sprintf("%v", obj[h]))
										}
										csvWriter.Write(row)
									}
								}
							}
							csvWriter.Flush()
							file.Close()
							fileInfo, _ := os.Stat(filepath)
							uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
							DB.Create(&uploadedFile)
							json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": len(arr), "columns": 0, "size": fileInfo.Size()})
							return
						}
					}
				}
			}
			http.Error(w, "Could not parse API response", http.StatusInternalServerError)
			return
		}

	case "graphql":
		if conn.Endpoint != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			gqlQuery := fmt.Sprintf(`{"query":"{ %s { id } }"}`, input.TableName)
			req, _ := http.NewRequest("POST", conn.Endpoint, strings.NewReader(gqlQuery))
			req.Header.Set("Content-Type", "application/json")
			if conn.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			}
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "GraphQL query failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(resp.Body)
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("graphql_%s.json", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": 0, "columns": 0, "size": fileInfo.Size()})
			return
		}

	case "google_drive", "google-drive":
		if conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			// Export as CSV
			exportURL := "https://www.googleapis.com/drive/v3/files/" + input.TableName + "/export?mimeType=text/csv"
			req, _ := http.NewRequest("GET", exportURL, nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "Google Drive export failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 50*1024*1024))
			fileID := generateSessionID()[:16]
			filename := fmt.Sprintf("gdrive_%s.csv", sanitizeTableName(input.TableName))
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			rowCount := strings.Count(string(bodyBytes), "\n")
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": rowCount, "columns": 0, "size": fileInfo.Size()})
			return
		}

	case "aws_s3", "aws-s3":
		if conn.Bucket != "" && conn.APIKey != "" {
			region := conn.Region
			if region == "" {
				region = "us-east-1"
			}
			httpClient := &http.Client{Timeout: 30 * time.Second}
			objURL := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", conn.Bucket, region, input.TableName)
			req, _ := http.NewRequest("GET", objURL, nil)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "S3 download failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 50*1024*1024))
			fileID := generateSessionID()[:16]
			filename := sanitizeFilename(input.TableName)
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			rowCount := strings.Count(string(bodyBytes), "\n")
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": rowCount, "columns": 0, "size": fileInfo.Size()})
			return
		}

	case "gcs":
		if conn.Bucket != "" && conn.APIKey != "" {
			httpClient := &http.Client{Timeout: 30 * time.Second}
			objURL := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media", conn.Bucket, input.TableName)
			req, _ := http.NewRequest("GET", objURL, nil)
			req.Header.Set("Authorization", "Bearer "+conn.APIKey)
			resp, err := httpClient.Do(req)
log.Printf("🔍 Chroma response: err=%v statusCode=%d", err, func() int { if resp != nil { return resp.StatusCode }; return 0 }())
			if err != nil {
				http.Error(w, "GCS download failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			defer resp.Body.Close()
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 50*1024*1024))
			fileID := generateSessionID()[:16]
			filename := sanitizeFilename(input.TableName)
			filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)
			os.WriteFile(filepath, bodyBytes, 0644)
			fileInfo, _ := os.Stat(filepath)
			rowCount := strings.Count(string(bodyBytes), "\n")
			uploadedFile := UploadedFile{ID: fileID, Filename: filename, Path: filepath, Size: fileInfo.Size(), UserID: session.UserID, CreatedAt: time.Now()}
			DB.Create(&uploadedFile)
			json.NewEncoder(w).Encode(map[string]interface{}{"file_id": fileID, "filename": filename, "rows": rowCount, "columns": 0, "size": fileInfo.Size()})
			return
		}

	default:
		http.Error(w, "Unsupported database type", http.StatusBadRequest)
		return
	}
	defer rows.Close()

	// Get column names
	columns, _ := rows.Columns()

	// Create CSV file
	fileID := generateSessionID()[:16]
	filename := fmt.Sprintf("%s_%s.csv", sanitizeFilename(conn.Database), sanitizeTableName(input.TableName))
	filepath := fmt.Sprintf("./uploads/%s_%s", fileID, filename)

	file, err := os.Create(filepath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write(columns)

	// Write rows
	rowCount := 0
	values := make([]interface{}, len(columns))
	valuePtrs := make([]interface{}, len(columns))
	for i := range values {
		valuePtrs[i] = &values[i]
	}

	for rows.Next() {
		rows.Scan(valuePtrs...)
		row := make([]string, len(columns))
		for i, v := range values {
			if v == nil {
				row[i] = ""
			} else {
				row[i] = fmt.Sprintf("%v", v)
			}
		}
		writer.Write(row)
		rowCount++
	}

	// Get file size
	fileInfo, _ := os.Stat(filepath)
	fileSize := fileInfo.Size()

	// Save to database
	uploadedFile := UploadedFile{
		ID:        fileID,
		Filename:  filename,
		Path:      filepath,
		Size:      fileSize,
		UserID:    session.UserID,
		CreatedAt: time.Now(),
	}
	DB.Create(&uploadedFile)

	json.NewEncoder(w).Encode(map[string]interface{}{
		"file_id":  fileID,
		"filename": filename,
		"rows":     rowCount,
		"columns":  len(columns),
		"size":     fileSize,
		"source":   conn.Name,
		"table":    input.TableName,
	})
}

// Google Drive OAuth Configuration
var googleOAuthConfig *oauth2.Config

func InitGoogleOAuth() {
	googleOAuthConfig = &oauth2.Config{
		ClientID:     os.Getenv("GOOGLE_CLIENT_ID"),
		ClientSecret: os.Getenv("GOOGLE_CLIENT_SECRET"),
		RedirectURL:  GetBaseURL() + "/api/google/callback",
		Scopes: []string{
			"https://www.googleapis.com/auth/drive.readonly",
		},
		Endpoint: google.Endpoint,
	}
}

// Google OAuth Start - redirects user to Google login
func GoogleAuthHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	// Store user session in state for callback
	state := session.UserID
	url := googleOAuthConfig.AuthCodeURL(state, oauth2.AccessTypeOffline)
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

// Google OAuth Callback - handles the redirect from Google
func GoogleCallbackHandler(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")
	userID := r.URL.Query().Get("state")

	if code == "" || userID == "" {
		http.Error(w, "Invalid callback", http.StatusBadRequest)
		return
	}

	token, err := googleOAuthConfig.Exchange(context.Background(), code)
	if err != nil {
		http.Error(w, "Token exchange failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Save token to database
	tokenJSON, _ := json.Marshal(token)
	conn := Connection{
		ID:        generateSessionID()[:16],
		Name:      "Google Drive",
		Type:      "cloud",
		SubType:   "google_drive",
		APIKey:    string(tokenJSON), // Store token as JSON
		Status:    "active",
		UserID:    userID,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	DB.Create(&conn)

	// Redirect back to data sources page
	http.Redirect(w, r, "/data-sources?google=connected", http.StatusTemporaryRedirect)
}

// List Google Drive files
func GoogleDriveListHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	connID := r.URL.Query().Get("connection_id")
	if connID == "" {
		http.Error(w, "Missing connection_id", http.StatusBadRequest)
		return
	}

	var conn Connection
	if err := DB.Where("id = ? AND user_id = ?", connID, session.UserID).First(&conn).Error; err != nil {
		http.Error(w, "Connection not found", http.StatusNotFound)
		return
	}

	// Parse stored token
	var token oauth2.Token
	if err := json.Unmarshal([]byte(conn.APIKey), &token); err != nil {
		http.Error(w, "Invalid token", http.StatusInternalServerError)
		return
	}

	// Create Drive client
	client := googleOAuthConfig.Client(context.Background(), &token)
	srv, err := drive.New(client)
	if err != nil {
		http.Error(w, "Failed to create Drive client: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// List spreadsheets and CSV files
	query := "mimeType='application/vnd.google-apps.spreadsheet' or mimeType='text/csv' or mimeType='application/vnd.ms-excel'"
	fileList, err := srv.Files.List().Q(query).Fields("files(id, name, mimeType, size)").Do()
	if err != nil {
		http.Error(w, "Failed to list files: "+err.Error(), http.StatusInternalServerError)
		return
	}

	files := make([]map[string]interface{}, len(fileList.Files))
	for i, f := range fileList.Files {
		files[i] = map[string]interface{}{
			"id":       f.Id,
			"name":     f.Name,
			"mimeType": f.MimeType,
			"size":     f.Size,
		}
	}

	json.NewEncoder(w).Encode(map[string]interface{}{"files": files})
}

// Generate API Key
func generateAPIKey() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return "sk-" + hex.EncodeToString(bytes)
}

// Create API Key Handler
func CreateAPIKeyHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	var input struct {
		Name           string   `json:"name"`
		Permissions    []string `json:"permissions"`
		RateLimit      string   `json:"rate_limit"`
		FineTunedModel string   `json:"finetuned_model"`
		LLMProvider    string   `json:"llm_provider"`
		LLMModel       string   `json:"llm_model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if input.Name == "" {
		input.Name = "Default API Key"
	}
	if input.RateLimit == "" {
		input.RateLimit = "1000/min"
	}
	permissions := "read,query"
	if len(input.Permissions) > 0 {
		permissions = strings.Join(input.Permissions, ",")
	}

	key := generateAPIKey()
	keyHash, _ := bcrypt.GenerateFromPassword([]byte(key), bcrypt.DefaultCost)

	apiKey := APIKey{
		ID:             generateSessionID()[:24],
		Name:           input.Name,
		Key:            key,
		KeyHash:        string(keyHash),
		UserID:         session.UserID,
		Permissions:    permissions,
		RateLimit:      input.RateLimit,
		Requests:       0,
		FineTunedModel: input.FineTunedModel,
		LLMProvider:    input.LLMProvider,
		LLMModel:       input.LLMModel,
		CreatedAt:      time.Now(),
	}
	DB.Create(&apiKey)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":          apiKey.ID,
		"name":        apiKey.Name,
		"key":         key,
		"permissions": strings.Split(apiKey.Permissions, ","),
		"rate_limit":  apiKey.RateLimit,
		"created_at":  apiKey.CreatedAt,
	})

}

// List API Keys Handler
func ListAPIKeysHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	var keys []APIKey
	DB.Where("user_id = ?", session.UserID).Order("created_at DESC").Find(&keys)

	result := make([]map[string]interface{}, len(keys))
	for i, k := range keys {
		perms := []string{}
		if k.Permissions != "" {
			perms = strings.Split(k.Permissions, ",")
		}
		result[i] = map[string]interface{}{
			"id":              k.ID,
			"name":            k.Name,
			"key":             k.Key,
			"permissions":     perms,
			"rate_limit":      k.RateLimit,
			"requests":        k.Requests,
			"finetuned_model": k.FineTunedModel,
			"llm_provider":    k.LLMProvider,
			"llm_model":       k.LLMModel,
			"last_used":       k.LastUsed,
			"created_at":      k.CreatedAt,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"keys": result})
}

// Delete API Key Handler
func DeleteAPIKeyHandler(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("session")
	if err != nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}
	session, _ := GetSession(cookie.Value)
	if session == nil {
		http.Error(w, "Not authenticated", http.StatusUnauthorized)
		return
	}

	keyID := r.URL.Query().Get("id")
	if keyID == "" {
		http.Error(w, "Missing key id", http.StatusBadRequest)
		return
	}

	result := DB.Where("id = ? AND user_id = ?", keyID, session.UserID).Delete(&APIKey{})
	if result.RowsAffected == 0 {
		http.Error(w, "Key not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"success": true})
}

// API Key Auth Middleware - for external API calls
func APIKeyAuthMiddleware(requiredPermission string) func(http.HandlerFunc) http.HandlerFunc {
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
						authHeader := r.Header.Get("Authorization")
			if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
				http.Error(w, "Missing or invalid Authorization header", http.StatusUnauthorized)
				return
			}

			key := strings.TrimPrefix(authHeader, "Bearer ")

			var keys []APIKey
			DB.Find(&keys)

			var validKey *APIKey
			for _, k := range keys {
				if bcrypt.CompareHashAndPassword([]byte(k.KeyHash), []byte(key)) == nil {
					validKey = &k
					break
				}
			}

			if validKey == nil {
				http.Error(w, "Invalid API key", http.StatusUnauthorized)
				return
			}

			// Check permission
			if requiredPermission != "" {
				hasPermission := false
				for _, p := range strings.Split(validKey.Permissions, ",") {
					if strings.TrimSpace(p) == requiredPermission {
						hasPermission = true
						break
					}
				}
				if !hasPermission {
					http.Error(w, "Permission denied: requires "+requiredPermission, http.StatusForbidden)
					return
				}
			}

			// Check fine-tuned model required
			if validKey.FineTunedModel == "" || validKey.FineTunedModel == "none" {
				http.Error(w, "Fine-tuned model required for API access", http.StatusForbidden)
				return
			}
			// Check rate limit
			if !checkRateLimit(validKey.ID, validKey.RateLimit) {
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}

			// Update last used and increment request count
			now := time.Now()
			DB.Model(&APIKey{}).Where("id = ?", validKey.ID).Updates(map[string]interface{}{
				"last_used": now,
				"requests":  gorm.Expr("requests + 1"),
			})

			r.Header.Set("X-User-ID", validKey.UserID)
			r.Header.Set("X-API-Key-ID", validKey.ID)
			r.Header.Set("X-LLM-Provider", validKey.LLMProvider)
			r.Header.Set("X-LLM-Model", validKey.LLMModel)
			r.Header.Set("X-FineTuned-Model", validKey.FineTunedModel)
			r.Header.Set("X-Rate-Limit", validKey.RateLimit)
			next(w, r)
		}
	}
}

// Rate limiting
var (
	rateLimitMutex sync.Mutex
	rateLimitMap   = make(map[string][]time.Time) // keyID -> request timestamps
)

func checkRateLimit(keyID string, limit string) bool {
	if limit == "" || limit == "unlimited" {
		return true
	}

	fmt.Printf("Rate limit check: keyID=%s, limit=%s", keyID, limit)
	// Parse limit like "1000/min"
	var maxRequests int
	fmt.Sscanf(limit, "%d/min", &maxRequests)
	if maxRequests == 0 {
		return true
	}

	rateLimitMutex.Lock()
	defer rateLimitMutex.Unlock()

	now := time.Now()
	windowStart := now.Add(-time.Minute)

	// Get existing timestamps and filter old ones
	timestamps := rateLimitMap[keyID]
	var validTimestamps []time.Time
	for _, t := range timestamps {
		if t.After(windowStart) {
			validTimestamps = append(validTimestamps, t)
		}
	}

	// Check if under limit
	fmt.Printf("Rate limit: maxRequests=%d, currentCount=%d", maxRequests, len(validTimestamps))
	if len(validTimestamps) >= maxRequests {
		rateLimitMap[keyID] = validTimestamps
		return false
	}

	// Add current request
	validTimestamps = append(validTimestamps, now)
	rateLimitMap[keyID] = validTimestamps
	return true
}

var emailService *EmailService

func initEmailService() {
	emailService = NewEmailService()
}

// Email validation
func isValidEmail(email string) bool {
	if email == "" || strings.Contains(email, " ") {
		return false
	}
	parts := strings.Split(email, "@")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return false
	}
	if !strings.Contains(parts[1], ".") {
		return false
	}
	return true
}

// Generate random 6-digit code
func generateVerificationCode() string {
	b := make([]byte, 3)
	rand.Read(b)
	return fmt.Sprintf("%06d", int(b[0])*10000+int(b[1])*100+int(b[2])%100)[:6]
}

// Send verification code for signup
func SendVerificationCodeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email string `json:"email"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request"})
		return
	}

	// Validate email
	req.Email = strings.TrimSpace(req.Email)
	if !isValidEmail(req.Email) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid email address"})
		return
	}

	// Check if email already exists
	var existingUser User
	if DB.Where("email = ?", req.Email).First(&existingUser).Error == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Email already registered"})
		return
	}

	// Generate code
	code := generateVerificationCode()

	// Delete old codes for this email
	DB.Where("email = ?", req.Email).Delete(&VerificationCode{})

	// Save new code
	verification := VerificationCode{
		Email:     req.Email,
		Code:      code,
		ExpiresAt: time.Now().Add(10 * time.Minute),
		CreatedAt: time.Now(),
	}
	DB.Create(&verification)

	// Initialize email service if needed
	if emailService == nil {
		initEmailService()
	}

	// Send email
	if err := emailService.SendVerificationCode(req.Email, code); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to send email: " + err.Error()})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Verification code sent"})
}

// Verify code and complete signup
func VerifyAndSignupHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email    string `json:"email"`
		Code     string `json:"code"`
		Name     string `json:"name"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request"})
		return
	}

	// Validate fields
	req.Email = strings.TrimSpace(req.Email)
	req.Name = strings.TrimSpace(req.Name)
	req.Code = strings.TrimSpace(req.Code)

	if !isValidEmail(req.Email) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid email address"})
		return
	}
	if req.Name == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Name is required"})
		return
	}
	if len(req.Password) < 6 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Password must be at least 6 characters"})
		return
	}
	if len(req.Code) != 6 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid verification code"})
		return
	}

	// Find verification code
	var verification VerificationCode
	if err := DB.Where("email = ? AND code = ? AND used = ? AND expires_at > ?",
		req.Email, req.Code, false, time.Now()).First(&verification).Error; err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid or expired verification code"})
		return
	}

	// Mark code as used
	verification.Used = true
	DB.Save(&verification)

	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to hash password"})
		return
	}

	// Create user
	user := User{
		ID:        generateSessionID(),
		Name:      req.Name,
		Email:     req.Email,
		Password:  string(hashedPassword),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := DB.Create(&user).Error; err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to create user"})
		return
	}

	// Create session
	session := Session{
		UserID:    user.ID,
		Email:     user.Email,
		Name:      user.Name,
		ExpiresAt: time.Now().Add(7 * 24 * time.Hour),
	}

	token := generateSessionID()
	sessionJSON, _ := json.Marshal(session)
	rdb.Set(context.Background(), "session:"+token, sessionJSON, 7*24*time.Hour)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"user":  user,
		"token": token,
	})
}

// Request password reset - sends verification code
func RequestPasswordResetHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email string `json:"email"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request"})
		return
	}

	// Validate email
	req.Email = strings.TrimSpace(req.Email)
	if !isValidEmail(req.Email) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid email address"})
		return
	}

	// Find user
	var user User
	if err := DB.Where("email = ?", req.Email).First(&user).Error; err != nil {
		// Don't reveal if email exists - but still return success
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "If the email exists, a reset code will be sent"})
		return
	}

	// Generate verification code
	code := generateVerificationCode()

	// Delete old codes for this email (reuse VerificationCode table)
	DB.Where("email = ?", req.Email).Delete(&VerificationCode{})

	// Save new code
	verification := VerificationCode{
		Email:     req.Email,
		Code:      code,
		ExpiresAt: time.Now().Add(10 * time.Minute),
		CreatedAt: time.Now(),
	}
	DB.Create(&verification)

	// Initialize email service if needed
	if emailService == nil {
		initEmailService()
	}

	// Send email with code
	if err := emailService.SendPasswordResetCode(req.Email, code); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to send email"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Reset code sent to your email"})
}

// Verify reset code
func VerifyResetCodeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email string `json:"email"`
		Code  string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request"})
		return
	}

	// Find verification code
	var verification VerificationCode
	if err := DB.Where("email = ? AND code = ? AND used = ? AND expires_at > ?",
		req.Email, req.Code, false, time.Now()).First(&verification).Error; err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid or expired code"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Code verified"})
}

// Reset password with code
func ResetPasswordHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Email    string `json:"email"`
		Code     string `json:"code"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request"})
		return
	}

	// Validate fields
	req.Email = strings.TrimSpace(req.Email)
	req.Code = strings.TrimSpace(req.Code)

	if !isValidEmail(req.Email) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid email address"})
		return
	}
	if len(req.Password) < 6 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Password must be at least 6 characters"})
		return
	}
	if len(req.Code) != 6 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid verification code"})
		return
	}

	// Find and verify code
	var verification VerificationCode
	if err := DB.Where("email = ? AND code = ? AND used = ? AND expires_at > ?",
		req.Email, req.Code, false, time.Now()).First(&verification).Error; err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid or expired code"})
		return
	}

	// Mark code as used
	verification.Used = true
	DB.Save(&verification)

	// Find user
	var user User
	if err := DB.Where("email = ?", req.Email).First(&user).Error; err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	// Hash new password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to hash password"})
		return
	}

	// Update user password
	DB.Model(&User{}).Where("id = ?", user.ID).Update("password", string(hashedPassword))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Password reset successfully"})
}

// Google Login OAuth config (separate from Drive)
var googleLoginConfig *oauth2.Config

func initGoogleLoginConfig() {
	googleLoginConfig = &oauth2.Config{
		ClientID:     os.Getenv("GOOGLE_CLIENT_ID"),
		ClientSecret: os.Getenv("GOOGLE_CLIENT_SECRET"),
		RedirectURL:  GetBaseURL() + "/api/google/login/callback",
		Scopes: []string{
			"https://www.googleapis.com/auth/userinfo.email",
			"https://www.googleapis.com/auth/userinfo.profile",
		},
		Endpoint: google.Endpoint,
	}
}

// Google Login Start
func GoogleLoginHandler(w http.ResponseWriter, r *http.Request) {
	if googleLoginConfig == nil {
		initGoogleLoginConfig()
	}

	state := generateSessionID()
	// Store state in Redis for verification
	rdb.Set(context.Background(), "google_state:"+state, "pending", 10*time.Minute)

	url := googleLoginConfig.AuthCodeURL(state, oauth2.AccessTypeOffline)
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

// Google Login Callback
func GoogleLoginCallbackHandler(w http.ResponseWriter, r *http.Request) {
	if googleLoginConfig == nil {
		initGoogleLoginConfig()
	}

	code := r.URL.Query().Get("code")
	state := r.URL.Query().Get("state")

	if code == "" || state == "" {
		http.Redirect(w, r, "/login?error=invalid_callback", http.StatusTemporaryRedirect)
		return
	}

	// Verify state
	val, err := rdb.Get(context.Background(), "google_state:"+state).Result()
	if err != nil || val != "pending" {
		http.Redirect(w, r, "/login?error=invalid_state", http.StatusTemporaryRedirect)
		return
	}
	rdb.Del(context.Background(), "google_state:"+state)

	// Exchange code for token
	token, err := googleLoginConfig.Exchange(context.Background(), code)
	if err != nil {
		http.Redirect(w, r, "/login?error=token_exchange_failed", http.StatusTemporaryRedirect)
		return
	}

	// Get user info from Google
	client := googleLoginConfig.Client(context.Background(), token)
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		http.Redirect(w, r, "/login?error=userinfo_failed", http.StatusTemporaryRedirect)
		return
	}
	defer resp.Body.Close()

	var googleUser struct {
		ID      string `json:"id"`
		Email   string `json:"email"`
		Name    string `json:"name"`
		Picture string `json:"picture"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&googleUser); err != nil {
		http.Redirect(w, r, "/login?error=decode_failed", http.StatusTemporaryRedirect)
		return
	}

	// Check if user exists
	var user User
	if err := DB.Where("email = ?", googleUser.Email).First(&user).Error; err != nil {
		// Create new user
		user = User{
			ID:        generateSessionID(),
			Name:      googleUser.Name,
			Email:     googleUser.Email,
			Image:     googleUser.Picture,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		DB.Create(&user)
	} else {
		// Update user info if needed
		if user.Image == "" && googleUser.Picture != "" {
			user.Image = googleUser.Picture
			DB.Save(&user)
		}
	}

	// Create session
	sessionToken := generateSessionID()
	session := Session{
		UserID:    user.ID,
		Email:     user.Email,
		Name:      user.Name,
		ExpiresAt: time.Now().Add(7 * 24 * time.Hour),
	}
	sessionJSON, _ := json.Marshal(session)
	rdb.Set(context.Background(), "session:"+sessionToken, sessionJSON, 7*24*time.Hour)

	// Set cookie and redirect
	http.SetCookie(w, &http.Cookie{
		Name:     "session",
		Value:    sessionToken,
		Path:     "/",
		MaxAge:   7 * 24 * 60 * 60,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})

	http.Redirect(w, r, "/", http.StatusTemporaryRedirect)
}

// UpdateProfileHandler updates user profile
func UpdateProfileHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Name         string `json:"name"`
		Organization string `json:"organization"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Update user
	result := DB.Model(&User{}).Where("id = ?", userID).Select("name", "updated_at").Updates(User{Name: strings.TrimSpace(req.Name), UpdatedAt: time.Now()})

	if result.Error != nil {
		http.Error(w, "Failed to update profile", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Profile updated successfully",
	})
}

// DeleteAccountHandler deletes user account
func DeleteAccountHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Delete user's data
	DB.Where("user_id = ?", userID).Delete(&Query{})
	DB.Where("user_id = ?", userID).Delete(&FineTunedModel{})
	DB.Where("id = ?", userID).Delete(&User{})

	// Clear session
	cookie := &http.Cookie{
		Name:     "session_id",
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
	}
	http.SetCookie(w, cookie)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Account deleted successfully",
	})
}

// ChangePasswordRequestHandler - sends verification code for password change
func ChangePasswordRequestHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	code := fmt.Sprintf("%06d", time.Now().UnixNano()%1000000)

	ctx := context.Background()
	rdb.Set(ctx, "password_change:"+user.Email, code, 10*time.Minute)

	if emailService == nil {
		http.Error(w, "Email service not configured", http.StatusInternalServerError)
		return
	}

	if err := emailService.SendPasswordResetCode(user.Email, code); err != nil {
		http.Error(w, "Failed to send email", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Verification code sent to your email",
	})
}

// ChangePasswordVerifyHandler - verifies code and changes password
func ChangePasswordVerifyHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Code        string `json:"code"`
		NewPassword string `json:"new_password"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if len(req.NewPassword) < 6 {
		http.Error(w, "Password must be at least 6 characters", http.StatusBadRequest)
		return
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	ctx := context.Background()
	storedCode, err := rdb.Get(ctx, "password_change:"+user.Email).Result()
	if err != nil || storedCode != req.Code {
		http.Error(w, "Invalid or expired code", http.StatusBadRequest)
		return
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.NewPassword), bcrypt.DefaultCost)
	if err != nil {
		http.Error(w, "Failed to hash password", http.StatusInternalServerError)
		return
	}

	DB.Model(&User{}).Where("id = ?", userID).Update("password", string(hashedPassword))
	rdb.Del(ctx, "password_change:"+user.Email)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Password changed successfully",
	})
}

// LogoutAllDevicesHandler - logs out from all devices
func LogoutAllDevicesHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	cookie, err := r.Cookie("session_id")
	currentSessionID := ""
	if err == nil {
		currentSessionID = cookie.Value
	}

	ctx := context.Background()
	iter := rdb.Scan(ctx, 0, "session:*", 0).Iterator()

	for iter.Next(ctx) {
		key := iter.Val()
		sessionData, _ := rdb.Get(ctx, key).Result()
		if strings.Contains(sessionData, userID) {
			sessionID := strings.TrimPrefix(key, "session:")
			if sessionID != currentSessionID {
				rdb.Del(ctx, key)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Logged out from all other devices",
	})
}

// UploadAvatarHandler - uploads user avatar
func UploadAvatarHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	err := r.ParseMultipartForm(1 << 30)
	if err != nil {
		http.Error(w, "File too large (max 5MB)", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("avatar")
	if err != nil {
		http.Error(w, "No file uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	contentType := header.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "image/") {
		http.Error(w, "Only image files allowed", http.StatusBadRequest)
		return
	}

	avatarDir := "./uploads/avatars"
	os.MkdirAll(avatarDir, 0755)

	ext := ".jpg"
	if strings.Contains(contentType, "png") {
		ext = ".png"
	} else if strings.Contains(contentType, "gif") {
		ext = ".gif"
	} else if strings.Contains(contentType, "webp") {
		ext = ".webp"
	}

	filename := fmt.Sprintf("%s%s", sanitizeFilename(userID), ext)
	filepath := fmt.Sprintf("%s/%s", avatarDir, filename)

	dst, err := os.Create(filepath)
	if err != nil {
		http.Error(w, "Failed to save file", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	_, err = io.Copy(dst, file)
	if err != nil {
		http.Error(w, "Failed to save file", http.StatusInternalServerError)
		return
	}

	avatarURL := fmt.Sprintf("/uploads/avatars/%s", filename)
	DB.Model(&User{}).Where("id = ?", userID).Update("avatar_url", avatarURL)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":    true,
		"avatar_url": avatarURL,
	})
}

// GetSessionsHandler - returns user's active sessions
func GetSessionsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Get current session ID
	cookie, _ := r.Cookie("session_id")
	currentSessionID := ""
	if cookie != nil {
		currentSessionID = cookie.Value
	}

	// For now, return mock data - in production, query from database
	sessions := []map[string]interface{}{
		{"id": "1", "device": "Chrome (Mac OS X)", "device_type": "desktop", "location": "Istanbul, TR", "created_at": time.Now().Add(-24 * time.Hour).Format(time.RFC3339), "updated_at": time.Now().Format(time.RFC3339), "is_current": true},
	}

	// Mark current session
	for i := range sessions {
		if sessions[i]["id"] == currentSessionID {
			sessions[i]["is_current"] = true
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sessions": sessions,
		"total":    len(sessions),
	})
}

// UsageLog model
type UsageLog struct {
	ID           string    `gorm:"primaryKey" json:"id"`
	UserID       string    `json:"user_id"`
	EventType    string    `json:"event_type"`
	EventName    string    `json:"event_name"`
	ResourceID   string    `json:"resource_id"`
	ResourceName string    `json:"resource_name"`
	CreditsUsed  float64   `json:"credits_used"`
	TokensUsed   int       `json:"tokens_used"`
	ModelUsed    string    `json:"model_used"`
	CreatedAt    time.Time `json:"created_at"`
}

func (UsageLog) TableName() string { return "usage_logs" }
