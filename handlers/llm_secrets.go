package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

var validLLMModels = map[string]map[string]bool{
	"openai": {
		"gpt-4o": true, "gpt-4o-mini": true, "gpt-4-turbo": true, "gpt-4": true,
		"gpt-3.5-turbo": true, "o1": true, "o1-mini": true, "o3-mini": true,
	},
	"anthropic": {
		"claude-sonnet-4-5": true, "claude-opus-4": true, "claude-opus-4-1": true,
		"claude-sonnet-4": true, "claude-haiku-4": true,
		"claude-3-5-sonnet-20241022": true, "claude-3-5-haiku-20241022": true,
		"claude-3-opus-20240229": true,
	},
	"gemini": {
		"gemini-2.5-flash": true, "gemini-2.5-pro": true,
		"gemini-2.0-flash": true, "gemini-2.0-flash-exp": true,
		"gemini-1.5-pro": true, "gemini-1.5-flash": true,
	},
	"mistral": {
		"mistral-large-latest": true, "mistral-medium-latest": true,
		"mistral-small-latest": true, "ministral-8b-latest": true,
		"ministral-3b-latest": true, "codestral-latest": true,
	},
}

const maxSelectedModels = 50

func UpdateLLMSecretModelsHandler(w http.ResponseWriter, r *http.Request) {
	writeJSONError := func(status int, code, message string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "error",
			"code":    code,
			"message": message,
		})
	}

	if r.Method != http.MethodPost {
		writeJSONError(http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}

	userID := strings.TrimSpace(r.Header.Get("X-User-ID"))
	if userID == "" {
		writeJSONError(http.StatusUnauthorized, "unauthorized", "Authentication required")
		return
	}

	var req struct {
		ID     string   `json:"id"`
		Models []string `json:"models"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 32*1024)).Decode(&req); err != nil {
		writeJSONError(http.StatusBadRequest, "invalid_json", "Request body is not valid JSON")
		return
	}

	req.ID = strings.TrimSpace(req.ID)
	if req.ID == "" {
		writeJSONError(http.StatusBadRequest, "missing_id", "Secret ID is required")
		return
	}
	if len(req.Models) == 0 {
		writeJSONError(http.StatusBadRequest, "empty_models", "At least one model must be selected")
		return
	}
	if len(req.Models) > maxSelectedModels {
		writeJSONError(http.StatusUnprocessableEntity, "too_many_models", fmt.Sprintf("Maximum %d models allowed per secret", maxSelectedModels))
		return
	}

	seen := make(map[string]bool, len(req.Models))
	cleaned := make([]string, 0, len(req.Models))
	for _, m := range req.Models {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		if len(m) > 200 {
			writeJSONError(http.StatusUnprocessableEntity, "invalid_model_name", "Model name exceeds 200 characters")
			return
		}
		if seen[m] {
			continue
		}
		seen[m] = true
		cleaned = append(cleaned, m)
	}
	if len(cleaned) == 0 {
		writeJSONError(http.StatusBadRequest, "empty_models", "At least one non-empty model required")
		return
	}

	var secret struct {
		ID       string
		UserID   string
		Provider string
	}
	if err := DB.Raw("SELECT id, user_id, provider FROM llm_secrets WHERE id = ?", req.ID).Scan(&secret).Error; err != nil {
		log.Printf("[LLM_SECRETS] lookup failed id=%s user=%s err=%v", req.ID, userID, err)
		writeJSONError(http.StatusInternalServerError, "db_error", "Database lookup failed")
		return
	}
	if secret.ID == "" {
		writeJSONError(http.StatusNotFound, "not_found", "Secret not found")
		return
	}
	if secret.UserID != userID {
		log.Printf("[LLM_SECRETS] ownership violation: user=%s tried to modify secret=%s owned by %s", userID, req.ID, secret.UserID)
		writeJSONError(http.StatusForbidden, "forbidden", "You do not own this secret")
		return
	}

	provider := strings.ToLower(strings.TrimSpace(secret.Provider))
	if allowedForProvider, hasProvider := validLLMModels[provider]; hasProvider {
		for _, m := range cleaned {
			if !allowedForProvider[m] {
				writeJSONError(http.StatusUnprocessableEntity, "invalid_model_for_provider", fmt.Sprintf("Model %q is not supported by provider %q", m, provider))
				return
			}
		}
	}

	modelsJSON, err := json.Marshal(cleaned)
	if err != nil {
		writeJSONError(http.StatusInternalServerError, "serialization_error", "Failed to serialize models")
		return
	}

	res := DB.Exec(
		"UPDATE llm_secrets SET selected_models = ?, updated_at = NOW() WHERE id = ? AND user_id = ?",
		string(modelsJSON), req.ID, userID,
	)
	if res.Error != nil {
		log.Printf("[LLM_SECRETS] update failed id=%s user=%s err=%v", req.ID, userID, res.Error)
		writeJSONError(http.StatusInternalServerError, "db_error", "Failed to update models")
		return
	}
	if res.RowsAffected == 0 {
		writeJSONError(http.StatusNotFound, "not_found", "Secret not found or already removed")
		return
	}

	log.Printf("[LLM_SECRETS] updated id=%s user=%s provider=%s models=%d", req.ID, userID, provider, len(cleaned))

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "success",
		"id":       req.ID,
		"provider": provider,
		"models":   cleaned,
		"count":    len(cleaned),
	})
}
