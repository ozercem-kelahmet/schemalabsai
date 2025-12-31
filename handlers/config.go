package handlers

import (
	"os"
	"strings"
)

// IsProduction auto-detects production environment
func IsProduction() bool {
	// Check common production indicators
	if os.Getenv("APP_ENV") == "production" || os.Getenv("NODE_ENV") == "production" {
		return true
	}
	// Check if running on production host
	if host := os.Getenv("HOSTNAME"); strings.Contains(host, "schemalabs") {
		return true
	}
	// Check if DATABASE_URL contains production host (not localhost)
	if db := os.Getenv("DATABASE_URL"); db != "" && !strings.Contains(db, "localhost") {
		return true
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
	return "postgresql://schemalabs:SchemaLabs2024!@localhost:5432/schemalabs"
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
		return "https://schemalabs.ai"
	}
	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}
	return "http://localhost:" + port
}
