package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
)

func isAdmin(r *http.Request) bool {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		return false
	}
	var role string
	err := DB.Raw("SELECT role FROM users WHERE id = ?", userID).Scan(&role).Error
	if err != nil {
		return false
	}
	return role == "admin"
}

func AdminUsersHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	if r.Method == "GET" {
		var users []map[string]interface{}
		DB.Raw("SELECT id, email, name, role, created_at FROM users ORDER BY created_at DESC").Scan(&users)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(users)
		return
	}

	if r.Method == "DELETE" {
		id := strings.TrimPrefix(r.URL.Path, "/api/admin/users/")
		id = strings.TrimSuffix(id, "/role")
		DB.Exec("DELETE FROM users WHERE id = ?", id)
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method == "PUT" && strings.Contains(r.URL.Path, "/role") {
		parts := strings.Split(strings.TrimSuffix(r.URL.Path, "/"), "/")
		id := parts[len(parts)-2]
		var body struct{ Role string `json:"role"` }
		json.NewDecoder(r.Body).Decode(&body)
		DB.Exec("UPDATE users SET role = ? WHERE id = ?", body.Role, id)
		w.WriteHeader(http.StatusOK)
		return
	}
}

func AdminModelsHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	if r.Method == "GET" {
		var models []map[string]interface{}
		DB.Raw(`SELECT m.id, m.name, m.accuracy, m.epochs, m.request_count, m.created_at, u.email as user_email
			FROM fine_tuned_models m LEFT JOIN users u ON m.user_id = u.id ORDER BY m.created_at DESC`).Scan(&models)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models)
		return
	}

	if r.Method == "DELETE" {
		id := strings.TrimPrefix(r.URL.Path, "/api/admin/models/")
		DB.Exec("DELETE FROM fine_tuned_models WHERE id = ?", id)
		w.WriteHeader(http.StatusOK)
		return
	}
}

func AdminKeysHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	if r.Method == "GET" {
		var keys []map[string]interface{}
		DB.Raw(`SELECT k.id, k.name, k.key, k.requests, k.created_at, u.email as user_email
			FROM api_keys k LEFT JOIN users u ON k.user_id = u.id ORDER BY k.created_at DESC`).Scan(&keys)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(keys)
		return
	}

	if r.Method == "DELETE" {
		id := strings.TrimPrefix(r.URL.Path, "/api/admin/keys/")
		DB.Exec("DELETE FROM api_keys WHERE id = ?", id)
		w.WriteHeader(http.StatusOK)
		return
	}
}

func AdminFilesHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	if r.Method == "GET" {
		var files []map[string]interface{}
		DB.Raw(`SELECT f.id, f.filename, f.size, f.created_at, u.email as user_email
			FROM uploaded_files f LEFT JOIN users u ON f.user_id = u.id ORDER BY f.created_at DESC`).Scan(&files)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(files)
		return
	}

	if r.Method == "DELETE" {
		id := strings.TrimPrefix(r.URL.Path, "/api/admin/files/")
		DB.Exec("DELETE FROM uploaded_files WHERE id = ?", id)
		w.WriteHeader(http.StatusOK)
		return
	}
}

func AdminQueriesHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	var queries []map[string]interface{}
	DB.Raw(`SELECT q.id, q.name, q.created_at, u.email as user_email
		FROM queries q LEFT JOIN users u ON q.user_id = u.id ORDER BY q.created_at DESC`).Scan(&queries)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(queries)
}

func AdminEndpointsHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	var endpoints []map[string]interface{}
	DB.Raw(`SELECT e.id, e.endpoint, e.model_name, e.status, e.created_at, u.email as user_email
		FROM endpoints e LEFT JOIN users u ON e.user_id = u.id ORDER BY e.created_at DESC`).Scan(&endpoints)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(endpoints)
}

func AdminConfigHandler(w http.ResponseWriter, r *http.Request) {
	if !isAdmin(r) {
		http.Error(w, "Unauthorized", http.StatusForbidden)
		return
	}

	if r.Method == "GET" {
		config := map[string]interface{}{
			"modelPath":    "./checkpoints/model_12sector.pt",
			"smtpEmail":    "hello@schemalabs.ai",
			"maxFileSize":  50,
			"maxStorage":   1024,
			"openaiKey":    "",
			"anthropicKey": "",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(config)
		return
	}

	if r.Method == "PUT" {
		w.WriteHeader(http.StatusOK)
		return
	}
}
