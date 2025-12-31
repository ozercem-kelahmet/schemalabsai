package handlers

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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

var DB *gorm.DB
var rdb *redis.Client
var ctx = context.Background()

type User struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	Name      string    `json:"name"`
	Email     string    `gorm:"unique" json:"email"`
	Password  string    `json:"-"`
	Image     string    `json:"image"`
	Role string `json:"role"`
	Plan string `json:"plan"`
	MaxTeams int `json:"max_teams"`
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
	DB.AutoMigrate(&User{}, &UploadedFile{}, &Query{}, &Message{}, &QueryFile{}, &FineTunedModel{}, &Folder{}, &Connection{}, &APIKey{}, &VerificationCode{}, &PasswordResetToken{})

	// Redis
	redisURL := GetRedisURL()
	rdb = redis.NewClient(&redis.Options{
		Addr:     redisURL,
		Password: os.Getenv("REDIS_PASSWORD"),
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
		http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	if bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(req.Password)) != nil {
		http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		return
	}

	sessionID, _ := CreateSession(user.ID, user.Email, user.Name)

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
	TrainingModelID *string   `json:"training_model_id"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

func (Query) TableName() string { return "queries" }

// Message model
type Message struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	Role      string    `json:"role"`
	Content   string    `gorm:"type:text" json:"content"`
	Model     string    `json:"model"`
	Tokens    int       `json:"tokens"`
	QueryID   string    `json:"query_id"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
}

// QueryFile - many to many
type QueryFile struct {
	QueryID string `gorm:"primaryKey" json:"query_id"`
	FileID  string `gorm:"primaryKey" json:"file_id"`
}

type FineTunedModel struct {
	ID           string    `gorm:"primaryKey" json:"id"`
	Name         string    `json:"name"`
	Version      int       `json:"version"`
	SourceFileID string    `json:"source_file_id"`
	SourceName   string    `json:"source_name"`
	ModelPath    string    `json:"model_path"`
	Accuracy     float64   `json:"accuracy"`
	Epochs       int       `json:"epochs"`
	BatchSize    int       `json:"batch_size"`
	Loss         float64   `json:"loss"`
	UserID       string    `json:"user_id"`
	CreatedAt    time.Time `json:"created_at"`
}

type Folder struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	Name      string    `json:"name"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
}

// Connection types
type Connection struct {
	ID           string     `gorm:"primaryKey" json:"id"`
	Name         string     `json:"name"`
	Type         string     `json:"type"`     // database, vectordb, cloud, api
	SubType      string     `json:"sub_type"` // postgresql, mysql, pinecone, etc.
	Host         string     `json:"host"`
	Port         int        `json:"port"`
	Database     string     `json:"database"`
	Username     string     `json:"username"`
	Password     string     `json:"-"`
	APIKey       string     `json:"-"`
	Endpoint     string     `json:"endpoint"`
	Bucket       string     `json:"bucket"`
	Region       string     `json:"region"`
	SSL          bool       `json:"ssl"`
	Status       string     `json:"status"` // active, error, disconnected
	LastTestedAt *time.Time `json:"last_tested_at"`
	UserID       string     `json:"user_id"`
	CreatedAt    time.Time  `json:"created_at"`
	UpdatedAt    time.Time  `json:"updated_at"`
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
			"created_at": c.CreatedAt,
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
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			input.Host, input.Port, input.Username, input.Password, input.Database)
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
		if input.Host != "" && input.Database != "" {
			mongoURI := fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", input.Username, input.Password, input.Host, input.Port, input.Database)
			if input.Username == "" {
				mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", input.Host, input.Port, input.Database)
			}
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
			message = "Host and database required"
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

	case "databricks":
		if input.Endpoint != "" && input.APIKey != "" {
			client := &http.Client{Timeout: 10 * time.Second}
			req, _ := http.NewRequest("GET", input.Endpoint+"/api/2.0/clusters/list", nil)
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

	var tables []string

	switch conn.SubType {
	case "postgresql", "supabase":
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			conn.Host, conn.Port, conn.Username, conn.Password, conn.Database)
		tempDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ := tempDB.DB()
		defer sqlDB.Close()

		rows, err := sqlDB.Query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		for rows.Next() {
			var name string
			rows.Scan(&name)
			tables = append(tables, name)
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
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer sfDB.Close()
		sfRows, err := sfDB.Query("SHOW TABLES")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer sfRows.Close()
		for sfRows.Next() {
			var createdOn, name, kind, dbName, schemaName string
			sfRows.Scan(&createdOn, &name, &kind, &dbName, &schemaName)
			tables = append(tables, name)
		}

	case "mongodb":
		mongoURI := fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		if conn.Username == "" {
			mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database)
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

	default:
		http.Error(w, "Unsupported database type", http.StatusBadRequest)
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{"tables": tables})
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
		dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			conn.Host, conn.Port, conn.Username, conn.Password, conn.Database)
		tempDB, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sqlDB, _ = tempDB.DB()
		defer sqlDB.Close()

		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", input.TableName, input.Limit)
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

		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", input.TableName, input.Limit)
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
		query := fmt.Sprintf("SELECT * FROM %s LIMIT %d", input.TableName, input.Limit)
		rows, err = sqlDB.Query(query)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

	case "mongodb":
		mongoURI := fmt.Sprintf("mongodb://%s:%s@%s:%d/%s", conn.Username, conn.Password, conn.Host, conn.Port, conn.Database)
		if conn.Username == "" {
			mongoURI = fmt.Sprintf("mongodb://%s:%d/%s", conn.Host, conn.Port, conn.Database)
		}
		clientOptions := options.Client().ApplyURI(mongoURI).SetConnectTimeout(10 * time.Second)
		mongoClient, err := mongo.Connect(context.Background(), clientOptions)
		if err != nil {
			http.Error(w, "Connection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer mongoClient.Disconnect(context.Background())

		collection := mongoClient.Database(conn.Database).Collection(input.TableName)
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
		filename := fmt.Sprintf("%s_%s.csv", conn.Database, input.TableName)
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

	default:
		http.Error(w, "Unsupported database type", http.StatusBadRequest)
		return
	}
	defer rows.Close()

	// Get column names
	columns, _ := rows.Columns()

	// Create CSV file
	fileID := generateSessionID()[:16]
	filename := fmt.Sprintf("%s_%s.csv", conn.Database, input.TableName)
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

// Email service instance
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

	http.Redirect(w, r, "/data-sources", http.StatusTemporaryRedirect)
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
	result := DB.Model(&User{}).Where("id = ?", userID).Updates(map[string]interface{}{
		"name":       req.Name,
		"updated_at": time.Now(),
	})

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

	err := r.ParseMultipartForm(5 << 20)
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

	filename := fmt.Sprintf("%s%s", userID, ext)
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
