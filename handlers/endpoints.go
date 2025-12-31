package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"os"
	"github.com/google/uuid"
)

type Endpoint struct {
	ID               string    `json:"id" gorm:"primaryKey"`
	UserID           string    `json:"user_id"`
	Name             string    `json:"name"`
	Path             string    `json:"path" gorm:"unique"`
	FineTunedModelID string    `json:"fine_tuned_model_id"`
	LLMModel         string    `json:"llm_model"`
	Description      string    `json:"description"`
	Calls            int       `json:"calls" gorm:"default:0"`
	Status           string    `json:"status" gorm:"default:active"`
	CreatedAt        time.Time `json:"created_at"`
}

func generateEndpointID() string {
	return uuid.New().String()[:16]
}

func ListEndpointsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var endpoints []Endpoint
	DB.Where("user_id = ?", userID).Order("created_at DESC").Find(&endpoints)

	if endpoints == nil {
		endpoints = []Endpoint{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(endpoints)
}

func CreateEndpointHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Name             string `json:"name"`
		Path             string `json:"path"`
		FineTunedModelID string `json:"fine_tuned_model_id"`
		LLMModel         string `json:"llm_model"`
		Description      string `json:"description"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Name == "" || req.Path == "" || req.FineTunedModelID == "" || req.LLMModel == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	req.Path = strings.ToLower(strings.TrimSpace(req.Path))
	req.Path = strings.ReplaceAll(req.Path, " ", "-")

	var count int64
	DB.Model(&Endpoint{}).Where("path = ?", req.Path).Count(&count)
	if count > 0 {
		http.Error(w, "Path already exists", http.StatusConflict)
		return
	}

	var model FineTunedModel
	if DB.Where("id = ? AND user_id = ?", req.FineTunedModelID, userID).First(&model).Error != nil {
		http.Error(w, "Invalid fine-tuned model", http.StatusBadRequest)
		return
	}

	endpoint := Endpoint{
		ID:               generateEndpointID(),
		UserID:           userID,
		Name:             req.Name,
		Path:             req.Path,
		FineTunedModelID: req.FineTunedModelID,
		LLMModel:         req.LLMModel,
		Description:      req.Description,
		Calls:            0,
		Status:           "active",
		CreatedAt:        time.Now(),
	}

	if err := DB.Create(&endpoint).Error; err != nil {
		http.Error(w, "Failed to create endpoint", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(endpoint)
}

func DeleteEndpointHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "DELETE" && r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		ID string `json:"id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	result := DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&Endpoint{})
	if result.RowsAffected == 0 {
		http.Error(w, "Endpoint not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func QueryEndpointHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/v1/query/")
	if path == "" {
		http.Error(w, "Invalid endpoint path", http.StatusBadRequest)
		return
	}

	authHeader := r.Header.Get("Authorization")
	if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
		http.Error(w, "Missing API key", http.StatusUnauthorized)
		return
	}
	apiKey := strings.TrimPrefix(authHeader, "Bearer ")

	var key APIKey
	if DB.Where("key = ?", apiKey).First(&key).Error != nil {
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	if !strings.Contains(key.Permissions, "query") {
		http.Error(w, "Permission denied: requires query", http.StatusForbidden)
		return
	}

	var endpoint Endpoint
	if DB.Where("path = ? AND user_id = ?", path, key.UserID).First(&endpoint).Error != nil {
		http.Error(w, "Endpoint not found", http.StatusNotFound)
		return
	}

	var req struct {
		Query  string                 `json:"query"`
		Data   map[string]interface{} `json:"data"`
		Format string                 `json:"format"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Update call counts
	DB.Model(&endpoint).Update("calls", endpoint.Calls+1)
	DB.Model(&key).Updates(map[string]interface{}{"requests": key.Requests + 1, "last_used": time.Now()})

	// Call fine-tuned model
	var fineTunedResult string
	if endpoint.FineTunedModelID != "" {
		queryWithData := req.Query
		if len(req.Data) > 0 {
			dataJSON, _ := json.Marshal(req.Data)
			queryWithData = req.Query + "\n\nInput Data: " + string(dataJSON)
		}
		result, err := callFineTunedModel(endpoint.FineTunedModelID, queryWithData)
		if err != nil {
			fmt.Printf("Fine-tuned model error: %v\n", err)
		} else {
			fineTunedResult = result
		}
	}

	// Build system prompt
	systemPrompt := "You are an AI assistant for data analysis. Provide clear, concise answers based on the model predictions and data analysis."
	if fineTunedResult != "" {
		systemPrompt += "\n\n### Fine-tuned Model Analysis:\n" + fineTunedResult + "\n\nUse this analysis to provide accurate insights."
	}

	// Prepare chat history
	history := []ChatMessage{{Role: "user", Content: req.Query}}

	// Determine which LLM to call based on endpoint config
	llmModel := endpoint.LLMModel
	var response string
	var tokens int
	var err error

	// Check model type and call appropriate API
	if isClaudeModel(llmModel) {
		response, tokens, err = callClaudeAPI(history, systemPrompt, llmModel, false, w)
	} else if isOpenAIModel(llmModel) {
		// Set header for OpenAI handler
		r.Header.Set("X-LLM-Model", llmModel)
		response, tokens, err = callOpenAIChat(history, systemPrompt, llmModel)
	} else {
		// Default to Claude
		response, tokens, err = callClaudeAPI(history, systemPrompt, "claude-3-5-sonnet-20241022", false, w)
	}

	if err != nil {
		http.Error(w, "LLM error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	result := map[string]interface{}{
		"answer":      response,
		"model":       llmModel,
		"fine_tuned":  endpoint.FineTunedModelID,
		"tokens_used": tokens,
		"endpoint_id": endpoint.ID,
	}

	if fineTunedResult != "" {
		result["prediction"] = fineTunedResult
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// Helper function to check if model is OpenAI
func isOpenAIModel(model string) bool {
	openAIModels := []string{"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"}
	for _, m := range openAIModels {
		if strings.Contains(strings.ToLower(model), m) {
			return true
		}
	}
	return false
}

// Call OpenAI API without streaming (for endpoint queries)
func callOpenAIChat(messages []ChatMessage, systemPrompt, model string) (string, int, error) {
	apiKey := getOpenAIKey()
	if apiKey == "" {
		return "", 0, fmt.Errorf("OpenAI API key not configured")
	}

	// Map model names
	modelMap := map[string]string{
		"gpt-4o":      "gpt-4o",
		"gpt-4o-mini": "gpt-4o-mini",
	}
	openAIModel, ok := modelMap[model]
	if !ok {
		openAIModel = "gpt-4o"
	}

	// Build messages array
	var openAIMessages []map[string]string
	openAIMessages = append(openAIMessages, map[string]string{
		"role":    "system",
		"content": systemPrompt,
	})
	for _, msg := range messages {
		openAIMessages = append(openAIMessages, map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	reqBody := map[string]interface{}{
		"model":       openAIModel,
		"messages":    openAIMessages,
		"max_tokens":  4096,
		"temperature": 0.7,
	}

	jsonBody, _ := json.Marshal(reqBody)
	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", strings.NewReader(string(jsonBody)))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()

	var openAIResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		return "", 0, err
	}

	if len(openAIResp.Choices) == 0 {
		return "", 0, fmt.Errorf("no response from OpenAI")
	}

	return openAIResp.Choices[0].Message.Content, openAIResp.Usage.TotalTokens, nil
}

// Helper to get OpenAI key
func getOpenAIKey() string {
	// Try environment variable first
	if key := strings.TrimSpace(getEnv("OPENAI_API_KEY", "")); key != "" {
		return key
	}
	return ""
}

func getEnv(key, defaultVal string) string {
	if val, ok := lookupEnv(key); ok {
		return val
	}
	return defaultVal
}

func lookupEnv(key string) (string, bool) {
	return os.LookupEnv(key)
	
}
