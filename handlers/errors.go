package handlers

import (
	"fmt"
	"net/http"
	"runtime/debug"
	"time"
)

type ErrorLog struct {
	ID         string    `gorm:"primaryKey" json:"id"`
	Level      string    `gorm:"default:error" json:"level"` // error, panic, warning
	Path       string    `json:"path"`
	Method     string    `json:"method"`
	StatusCode int       `json:"status_code"`
	Message    string    `json:"message"`
	StackTrace string    `json:"stack_trace"`
	UserID     string    `json:"user_id"`
	UserAgent  string    `json:"user_agent"`
	IP         string    `json:"ip"`
	CreatedAt  time.Time `json:"created_at"`
}

func InitErrorLogs() {
	DB.AutoMigrate(&ErrorLog{})
}

func LogError(level, path, method string, statusCode int, message, stackTrace, userID, userAgent, ip string) {
	log := ErrorLog{
		ID:         fmt.Sprintf("err-%d", time.Now().UnixNano()),
		Level:      level,
		Path:       path,
		Method:     method,
		StatusCode: statusCode,
		Message:    message,
		StackTrace: stackTrace,
		UserID:     userID,
		UserAgent:  userAgent,
		IP:         ip,
		CreatedAt:  time.Now(),
	}
	DB.Create(&log)
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok { f.Flush() }
}
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }
func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func RecoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		defer func() {
			if err := recover(); err != nil {
				stack := string(debug.Stack())
				ip := r.Header.Get("X-Forwarded-For")
				if ip == "" {
					ip = r.RemoteAddr
				}
				userID := r.Header.Get("X-User-ID")
				LogError("panic", r.URL.Path, r.Method, 500,
					fmt.Sprintf("%v", err), stack, userID, r.UserAgent(), ip)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(rec, r)
		if rec.status >= 500 {
			ip := r.Header.Get("X-Forwarded-For")
			if ip == "" {
				ip = r.RemoteAddr
			}
			userID := r.Header.Get("X-User-ID")
			LogError("error", r.URL.Path, r.Method, rec.status,
				fmt.Sprintf("HTTP %d", rec.status), "", userID, r.UserAgent(), ip)
		}
	}
}
