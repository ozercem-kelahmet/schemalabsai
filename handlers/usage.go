package handlers

import (
	"encoding/json"
	"net/http"
)

func GetUsageLogsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var logs []UsageLog
	if err := DB.Where("user_id = ?", userID).Order("created_at desc").Find(&logs).Error; err != nil {
		http.Error(w, "Failed to fetch usage logs", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"logs": logs,
	})
}
