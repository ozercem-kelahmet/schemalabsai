package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
)

// IsProduction auto-detects production environment
func IsProduction() bool {
	if os.Getenv("APP_ENV") == "production" || os.Getenv("NODE_ENV") == "production" {
		return true
	}
	if host := os.Getenv("HOSTNAME"); strings.Contains(host, "schemalabs") {
		return true
	}
	if _, err := os.Stat("/home/ozercemkelahmet/schemalabsai"); err == nil {
		if _, err := os.Stat("/etc/nginx/sites-enabled"); err == nil {
			return true
		}
	}
	return false
}

// GetFlaskURL returns Flask server URL based on environment
func GetFlaskURL() string {
	if url := os.Getenv("FLASK_URL"); url != "" {
		return url
	}
	if IsProduction() {
		return "https://api.schemalabs.ai"
	}
	port := os.Getenv("FLASK_PORT")
	if port == "" {
		port = "6000"
	}
	return "http://localhost:" + port
}

// GetDatabaseURL returns PostgreSQL connection string
func GetDatabaseURL() string {
	if dsn := os.Getenv("DATABASE_URL"); dsn != "" {
		return dsn
	}
	return ""
}

// GetRedisURL returns Redis connection string
func GetRedisURL() string {
	if url := os.Getenv("REDIS_URL"); url != "" {
		return url
	}
	return "localhost:6379"
}

// GetRedisPassword returns Redis password
func GetRedisPassword() string {
	return os.Getenv("REDIS_PASSWORD")
}

// GetBaseURL returns the base URL for callbacks
func GetBaseURL() string {
	if url := os.Getenv("BASE_URL"); url != "" {
		return url
	}
	if IsProduction() {
		return "https://console.schemalabs.ai"
	}
	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}
	return "http://localhost:" + port
}

// GetUploadLimitsHandler returns upload configuration
func GetUploadLimitsHandler(w http.ResponseWriter, r *http.Request) {
	maxFileSizeMB := getEnvInt("MAX_FILE_SIZE_MB", 50)
	maxTotalStorageMB := getEnvInt("MAX_TOTAL_STORAGE_MB", 1024)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"max_file_size_mb":     maxFileSizeMB,
		"max_total_storage_mb": maxTotalStorageMB,
	})
}
