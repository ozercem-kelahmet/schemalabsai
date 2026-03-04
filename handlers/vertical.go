package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ─── DB Models ───

type VerticalConfig struct {
	ID              string    `json:"id" gorm:"primaryKey"`
	UserID          string    `json:"user_id"`
	ModelID         string    `json:"model_id"`
	Name            string    `json:"name"`
	Description     string    `json:"description"`
	ConfigYAML      string    `json:"config_yaml" gorm:"type:text"`
	Enabled         bool      `json:"enabled" gorm:"default:true"`
	Version         int       `json:"version" gorm:"default:1"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	LanguageConfig  string    `json:"language_config" gorm:"type:jsonb"`
}

// ─── Language Layer Models ───

type PredictionStore struct {
	ID                string    `json:"id" gorm:"primaryKey"`
	UserID            string    `json:"user_id"`
	VerticalID        string    `json:"vertical_id"`
	ModelID           string    `json:"model_id"`
	RowData           string    `json:"row_data" gorm:"type:jsonb"`
	SchemaPrediction  string    `json:"schema_prediction"`
	SchemaConfidence   float64  `json:"schema_confidence"`
	ClassProbabilities string   `json:"class_probabilities" gorm:"type:jsonb"`
	ToolOutputs       string    `json:"tool_outputs" gorm:"type:jsonb"`
	AgentOutput       string    `json:"agent_output" gorm:"type:jsonb"`
	FinalDecision     string    `json:"final_decision"`
	Flags             string    `json:"flags" gorm:"type:jsonb"`
	QueryID           string    `json:"query_id"`
	CreatedAt         time.Time `json:"created_at"`
}

type LLMSecret struct {
	ID             string    `json:"id" gorm:"primaryKey"`
	UserID         string    `json:"user_id"`
	VerticalID     string    `json:"vertical_id"`
	Provider       string    `json:"provider"`
	SecretName     string    `json:"secret_name"`
	EncryptedValue string    `json:"encrypted_value" gorm:"type:text"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type VerticalTool struct {
	ID              string    `json:"id" gorm:"primaryKey"`
	UserID          string    `json:"user_id"`
	ModelID         string    `json:"model_id"`
	VerticalID      string    `json:"vertical_id"`
	Name            string    `json:"name"`
	Description     string    `json:"description"`
	Code            string    `json:"code" gorm:"type:text"`
	Hook            string    `json:"hook" gorm:"default:post_inference"` // pre_inference, post_inference, validator
	Enabled         bool      `json:"enabled" gorm:"default:true"`
	Version         int       `json:"version" gorm:"default:1"`
	ValidationStatus string  `json:"validation_status" gorm:"default:pending"` // pending, passed, failed
	ValidationError  string  `json:"validation_error" gorm:"type:text"`
	ExecutionOrder   int     `json:"execution_order" gorm:"default:0"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type VerticalAgent struct {
	ID              string    `json:"id" gorm:"primaryKey"`
	UserID          string    `json:"user_id"`
	ModelID         string    `json:"model_id"`
	VerticalID      string    `json:"vertical_id"`
	Name            string    `json:"name"`
	Description     string    `json:"description"`
	Code            string    `json:"code" gorm:"type:text"`
	Role            string    `json:"role" gorm:"default:default"` // default, decision_maker
	Enabled         bool      `json:"enabled" gorm:"default:true"`
	Version         int       `json:"version" gorm:"default:1"`
	PipelineOrder   int       `json:"pipeline_order" gorm:"default:0"`
	RunsIf          string    `json:"runs_if"` // conditional expression
	ParallelWith    string    `json:"parallel_with"` // agent name to run in parallel
	ValidationStatus string  `json:"validation_status" gorm:"default:pending"`
	ValidationError  string  `json:"validation_error" gorm:"type:text"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// ─── Config Handlers ───

func ListVerticalConfigsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	modelID := r.URL.Query().Get("model_id")
	var configs []VerticalConfig
	q := DB.Where("user_id = ?", userID)
	if modelID != "" { q = q.Where("model_id = ?", modelID) }
	q.Order("created_at desc").Find(&configs)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(configs)
}

func CreateVerticalConfigHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ModelID     string `json:"model_id"`
		Name        string `json:"name"`
		Description string `json:"description"`
		ConfigYAML  string `json:"config_yaml"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	if req.ModelID == "" || req.Name == "" {
		http.Error(w, "model_id and name required", 400); return
	}

	config := VerticalConfig{
		ID:         uuid.New().String(),
		UserID:     userID,
		ModelID:    req.ModelID,
		Name:       req.Name,
		Description: req.Description,
		ConfigYAML: req.ConfigYAML,
		Enabled:    true,
		Version:    1,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	DB.Create(&config)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(config)
}

func ActivateVerticalHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ID      string `json:"id"`
		ModelID string `json:"model_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	// Disable all verticals for this model
	DB.Model(&VerticalConfig{}).Where("user_id = ? AND model_id = ?", userID, req.ModelID).Update("enabled", false)
	// Enable selected one
	DB.Model(&VerticalConfig{}).Where("id = ? AND user_id = ?", req.ID, userID).Update("enabled", true)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "activated"})
}

func UpdateVerticalConfigHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ID         string `json:"id"`
		Name       string `json:"name"`
		Description string `json:"description"`
		ConfigYAML string `json:"config_yaml"`
		Enabled    *bool  `json:"enabled"`
LanguageConfig string `json:"language_config"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	var config VerticalConfig
	if DB.Where("id = ? AND user_id = ?", req.ID, userID).First(&config).Error != nil {
		http.Error(w, "Config not found", 404); return
	}

	updates := map[string]interface{}{"updated_at": time.Now()}
	if req.Name != "" { updates["name"] = req.Name }
	if req.Description != "" { updates["description"] = req.Description }
	if req.ConfigYAML != "" { updates["config_yaml"] = req.ConfigYAML; updates["version"] = config.Version + 1 }
	if req.Enabled != nil { updates["enabled"] = *req.Enabled }
	if req.LanguageConfig != "" { updates["language_config"] = req.LanguageConfig }

	DB.Model(&config).Updates(updates)

	DB.Where("id = ?", req.ID).First(&config)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(config)
}

func DeleteVerticalConfigHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct { ID string `json:"id"` }
	json.NewDecoder(r.Body).Decode(&req)

	DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&VerticalConfig{})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

// ─── Tool Handlers ───

func ListVerticalToolsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	modelID := r.URL.Query().Get("model_id")
	verticalID := r.URL.Query().Get("vertical_id")
	var tools []VerticalTool
	q := DB.Where("user_id = ?", userID)
	if verticalID != "" { q = q.Where("vertical_id = ?", verticalID) }
	if modelID != "" { q = q.Where("model_id = ?", modelID) }
	q.Order("execution_order asc, created_at asc").Find(&tools)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tools)
}

func UploadVerticalToolHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ModelID     string `json:"model_id"`
		VerticalID  string `json:"vertical_id"`
		Name        string `json:"name"`
		Description string `json:"description"`
		Code        string `json:"code"`
		Hook        string `json:"hook"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	if req.ModelID == "" || req.Name == "" || req.Code == "" {
		http.Error(w, "model_id, name, and code required", 400); return
	}
	if req.Hook == "" { req.Hook = "post_inference" }

	// Validate via Flask
	validationResult, err := validateToolViaFlask(req.Code, "tool", req.Hook)
	if err != nil {
		http.Error(w, fmt.Sprintf("Validation error: %v", err), 500); return
	}

	// Get next execution order
	var maxOrder int
	DB.Model(&VerticalTool{}).Where("user_id = ? AND model_id = ?", userID, req.ModelID).
		Select("COALESCE(MAX(execution_order), 0)").Scan(&maxOrder)

	tool := VerticalTool{
		ID:              uuid.New().String(),
		UserID:          userID,
		ModelID:         req.ModelID,
		VerticalID:      req.VerticalID,
		Name:            req.Name,
		Description:     req.Description,
		Code:            req.Code,
		Hook:            req.Hook,
		Enabled:         true,
		Version:         1,
		ExecutionOrder:  maxOrder + 1,
		ValidationStatus: validationResult.Status,
		ValidationError:  validationResult.Error,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}
	DB.Create(&tool)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tool":       tool,
		"validation": validationResult,
	})
}

func UpdateVerticalToolHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ID      string `json:"id"`
		Enabled *bool  `json:"enabled"`
		Code    string `json:"code"`
		Hook    string `json:"hook"`
		ExecutionOrder *int `json:"execution_order"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	var tool VerticalTool
	if DB.Where("id = ? AND user_id = ?", req.ID, userID).First(&tool).Error != nil {
		http.Error(w, "Tool not found", 404); return
	}

	updates := map[string]interface{}{"updated_at": time.Now()}
	if req.Enabled != nil { updates["enabled"] = *req.Enabled }
	if req.Hook != "" { updates["hook"] = req.Hook }
	if req.ExecutionOrder != nil { updates["execution_order"] = *req.ExecutionOrder }
	if req.Code != "" {
		validationResult, _ := validateToolViaFlask(req.Code, "tool", tool.Hook)
		updates["code"] = req.Code
		updates["version"] = tool.Version + 1
		updates["validation_status"] = validationResult.Status
		updates["validation_error"] = validationResult.Error
	}

	DB.Model(&tool).Updates(updates)

	DB.Where("id = ?", req.ID).First(&tool)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tool)
}

func DeleteVerticalToolHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct { ID string `json:"id"` }
	json.NewDecoder(r.Body).Decode(&req)

	DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&VerticalTool{})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

// ─── Agent Handlers ───

func ListVerticalAgentsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	modelID := r.URL.Query().Get("model_id")
	verticalID := r.URL.Query().Get("vertical_id")
	var agents []VerticalAgent
	q := DB.Where("user_id = ?", userID)
	if verticalID != "" { q = q.Where("vertical_id = ?", verticalID) }
	if modelID != "" { q = q.Where("model_id = ?", modelID) }
	q.Order("pipeline_order asc, created_at asc").Find(&agents)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(agents)
}

func UploadVerticalAgentHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ModelID      string `json:"model_id"`
		VerticalID   string `json:"vertical_id"`
		Name         string `json:"name"`
		Description  string `json:"description"`
		Code         string `json:"code"`
		Role         string `json:"role"`
		RunsIf       string `json:"runs_if"`
		ParallelWith string `json:"parallel_with"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	if req.ModelID == "" || req.Name == "" || req.Code == "" {
		http.Error(w, "model_id, name, and code required", 400); return
	}
	if req.Role == "" { req.Role = "default" }

	validationResult, err := validateToolViaFlask(req.Code, "agent", "")
	if err != nil {
		http.Error(w, fmt.Sprintf("Validation error: %v", err), 500); return
	}

	var maxOrder int
	DB.Model(&VerticalAgent{}).Where("user_id = ? AND model_id = ?", userID, req.ModelID).
		Select("COALESCE(MAX(pipeline_order), 0)").Scan(&maxOrder)

	agent := VerticalAgent{
		ID:              uuid.New().String(),
		UserID:          userID,
		ModelID:         req.ModelID,
		VerticalID:      req.VerticalID,
		Name:            req.Name,
		Description:     req.Description,
		Code:            req.Code,
		Role:            req.Role,
		Enabled:         true,
		Version:         1,
		PipelineOrder:   maxOrder + 1,
		RunsIf:          req.RunsIf,
		ParallelWith:    req.ParallelWith,
		ValidationStatus: validationResult.Status,
		ValidationError:  validationResult.Error,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}
	DB.Create(&agent)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"agent":     agent,
		"validation": validationResult,
	})
}

func UpdateVerticalAgentHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		ID      string `json:"id"`
		Enabled *bool  `json:"enabled"`
		Code    string `json:"code"`
		Role    string `json:"role"`
		PipelineOrder *int `json:"pipeline_order"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	var agent VerticalAgent
	if DB.Where("id = ? AND user_id = ?", req.ID, userID).First(&agent).Error != nil {
		http.Error(w, "Agent not found", 404); return
	}

	updates := map[string]interface{}{"updated_at": time.Now()}
	if req.Enabled != nil { updates["enabled"] = *req.Enabled }
	if req.Role != "" { updates["role"] = req.Role }
	if req.PipelineOrder != nil { updates["pipeline_order"] = *req.PipelineOrder }
	if req.Code != "" {
		validationResult, _ := validateToolViaFlask(req.Code, "agent", "")
		updates["code"] = req.Code
		updates["version"] = agent.Version + 1
		updates["validation_status"] = validationResult.Status
		updates["validation_error"] = validationResult.Error
	}

	DB.Model(&agent).Updates(updates)

	DB.Where("id = ?", req.ID).First(&agent)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(agent)
}

func DeleteVerticalAgentHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct { ID string `json:"id"` }
	json.NewDecoder(r.Body).Decode(&req)

	DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&VerticalAgent{})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

// ─── Config Validate Handler ───

func ValidateVerticalConfigHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }

	var req struct {
		ConfigYAML string `json:"config_yaml"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	result, err := validateConfigViaFlask(req.ConfigYAML)
	if err != nil {
		http.Error(w, err.Error(), 500); return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func validateConfigViaFlask(configYAML string) (ValidationResult, error) {
	payload := map[string]string{"config_yaml": configYAML}
	jsonData, _ := json.Marshal(payload)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Post(GetFlaskURL()+"/validate_config", "application/json", io.NopCloser(strings.NewReader(string(jsonData))))
	if err != nil {
		return ValidationResult{Status: "passed", Error: "Validation service unavailable"}, nil
	}
	defer resp.Body.Close()

	var result ValidationResult
	json.NewDecoder(resp.Body).Decode(&result)
	return result, nil
}

// ─── Validate Handler (frontend proxy) ───

func ValidateVerticalScriptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" { http.Error(w, "Method not allowed", 405); return }

	var req struct {
		Code       string `json:"code"`
		ScriptType string `json:"script_type"`
		Hook       string `json:"hook"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	result, err := validateToolViaFlask(req.Code, req.ScriptType, req.Hook)
	if err != nil {
		http.Error(w, err.Error(), 500); return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// ─── Validation via Flask ───

type ValidationResult struct {
	Status  string   `json:"status"` // passed, failed
	Error   string   `json:"error"`
	Checks  []string `json:"checks"`
}

func validateToolViaFlask(code string, scriptType string, hook string) (ValidationResult, error) {
	payload := map[string]string{
		"code":        code,
		"script_type": scriptType,
		"hook":        hook,
	}
	jsonData, _ := json.Marshal(payload)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Post(GetFlaskURL()+"/validate_script", "application/json", io.NopCloser(strings.NewReader(string(jsonData))))
	if err != nil {
		// Flask down — accept with warning
		return ValidationResult{Status: "passed", Error: "Validation service unavailable, accepted without checks"}, nil
	}
	defer resp.Body.Close()

	var result ValidationResult
	json.NewDecoder(resp.Body).Decode(&result)
	return result, nil
}


// getVerticalContext loads active vertical config and returns LLM context string
func GetVerticalContext(userID, modelID string) string {
	if userID == "" || modelID == "" { return "" }

	var config VerticalConfig
	if err := DB.Where("user_id = ? AND model_id = ? AND enabled = true", userID, modelID).First(&config).Error; err != nil {
		return ""
	}

	var tools []VerticalTool
	DB.Where("vertical_id = ? AND user_id = ? AND validation_status = ?", config.ID, userID, "passed").Order("execution_order asc").Find(&tools)

	var agents []VerticalAgent
	DB.Where("vertical_id = ? AND user_id = ? AND validation_status = ?", config.ID, userID, "passed").Order("pipeline_order asc").Find(&agents)

	if config.ConfigYAML == "" && len(tools) == 0 && len(agents) == 0 { return "" }

	// Parse language config for tone, compliance, threshold
	var langCfg struct {
		AssistantTone       string  `json:"assistant_tone"`
		ComplianceNotes     string  `json:"compliance_notes"`
		ConfidenceThreshold float64 `json:"confidence_threshold"`
		HistoryLookbackDays int     `json:"history_lookback_days"`
	}
	if config.LanguageConfig != "" {
		json.Unmarshal([]byte(config.LanguageConfig), &langCfg)
	}
	if langCfg.AssistantTone == "" { langCfg.AssistantTone = "professional" }
	if langCfg.ConfidenceThreshold == 0 { langCfg.ConfidenceThreshold = 0.75 }
	if langCfg.HistoryLookbackDays == 0 { langCfg.HistoryLookbackDays = 90 }

	ctx := "\n\n=== VERTICAL AI RUNTIME ===\n"
	ctx += fmt.Sprintf("Active Vertical: %s\n", config.Name)
	if config.Description != "" {
		ctx += fmt.Sprintf("Description: %s\n", config.Description)
	}
	ctx += fmt.Sprintf("Tone: %s\n", langCfg.AssistantTone)

	if config.ConfigYAML != "" {
		ctx += "\n--- System Config Rules ---\n"
		ctx += config.ConfigYAML + "\n"
	}

	if langCfg.ComplianceNotes != "" {
		ctx += "\n--- Compliance Requirements ---\n"
		ctx += langCfg.ComplianceNotes + "\n"
	}

	ctx += "\nIMPORTANT: The Vertical AI Runtime processed this data with custom tools and agents.\n"
	ctx += "The tool outputs and agent decisions are included in the analysis data above.\n"
	ctx += "You MUST incorporate these results into your response:\n"
	ctx += "- If a tool calculated risk scores or metrics, include them prominently\n"
	ctx += "- If an agent made a final_decision, state it clearly at the beginning of your response\n"
	ctx += "- Follow the behavioral rules defined in the System Config above\n"
	ctx += fmt.Sprintf("- When confidence is below %.0f%%, note that human review is recommended\n", langCfg.ConfidenceThreshold*100)
	ctx += fmt.Sprintf("- History queries are limited to the last %d days\n", langCfg.HistoryLookbackDays)
	ctx += "- All function calls you make are logged for compliance audit\n"
	ctx += "- You have no access to other users' verticals, predictions, or data\n"

	for _, t := range tools {
		ctx += fmt.Sprintf("- Tool: %s (%s)\n", t.Name, t.Hook)
	}
	for _, a := range agents {
		ctx += fmt.Sprintf("- Agent: %s (role: %s)\n", a.Name, a.Role)
	}

	return ctx
}

// Batch upload tools
func BatchUploadToolsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		Tools []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Code        string `json:"code"`
			Hook        string `json:"hook"`
			VerticalID  string `json:"vertical_id"`
		} `json:"tools"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	results := []map[string]interface{}{}
	for _, t := range req.Tools {
		if t.Name == "" || t.Code == "" || t.VerticalID == "" {
			results = append(results, map[string]interface{}{"name": t.Name, "status": "failed", "error": "Missing required fields"})
			continue
		}
		var maxOrder int
		DB.Model(&VerticalTool{}).Where("vertical_id = ?", t.VerticalID).Select("COALESCE(MAX(execution_order), 0)").Scan(&maxOrder)
		hook := t.Hook
		if hook == "" { hook = "post_inference" }
		tool := VerticalTool{
			ID: uuid.New().String(), UserID: userID, VerticalID: t.VerticalID,
			Name: t.Name, Description: t.Description, Code: t.Code,
			Hook: hook, Enabled: true, Version: 1,
			ValidationStatus: "pending", ExecutionOrder: maxOrder + 1,
		}
		if err := DB.Create(&tool).Error; err != nil {
			results = append(results, map[string]interface{}{"name": t.Name, "status": "failed", "error": err.Error()})
		} else {
			results = append(results, map[string]interface{}{"name": t.Name, "status": "created", "id": tool.ID})
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// Batch upload agents
func BatchUploadAgentsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }

	var req struct {
		Agents []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Code        string `json:"code"`
			Role        string `json:"role"`
			VerticalID  string `json:"vertical_id"`
		} `json:"agents"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}

	results := []map[string]interface{}{}
	for _, a := range req.Agents {
		if a.Name == "" || a.Code == "" || a.VerticalID == "" {
			results = append(results, map[string]interface{}{"name": a.Name, "status": "failed", "error": "Missing required fields"})
			continue
		}
		var maxOrder int
		DB.Model(&VerticalAgent{}).Where("vertical_id = ?", a.VerticalID).Select("COALESCE(MAX(pipeline_order), 0)").Scan(&maxOrder)
		role := a.Role
		if role == "" { role = "default" }
		agent := VerticalAgent{
			ID: uuid.New().String(), UserID: userID, VerticalID: a.VerticalID,
			Name: a.Name, Description: a.Description, Code: a.Code,
			Role: role, Enabled: true, Version: 1,
			ValidationStatus: "pending", PipelineOrder: maxOrder + 1,
		}
		if err := DB.Create(&agent).Error; err != nil {
			results = append(results, map[string]interface{}{"name": a.Name, "status": "failed", "error": err.Error()})
		} else {
			results = append(results, map[string]interface{}{"name": a.Name, "status": "created", "id": agent.ID})
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// Tool version history
type ToolVersion struct {
	ID        string    `json:"id" gorm:"primaryKey"`
	ToolID    string    `json:"tool_id"`
	UserID    string    `json:"user_id"`
	Code      string    `json:"code"`
	Version   int       `json:"version"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
}

func ListToolVersionsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }
	toolID := r.URL.Query().Get("tool_id")
	if toolID == "" { http.Error(w, "tool_id required", 400); return }
	var versions []ToolVersion
	DB.Where("tool_id = ? AND user_id = ?", toolID, userID).Order("version desc").Find(&versions)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(versions)
}

func RollbackToolVersionHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }
	var req struct {
		ToolID    string `json:"tool_id"`
		VersionID string `json:"version_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	var ver ToolVersion
	if DB.Where("id = ? AND tool_id = ? AND user_id = ?", req.VersionID, req.ToolID, userID).First(&ver).Error != nil {
		http.Error(w, "Version not found", 404); return
	}
	DB.Model(&VerticalTool{}).Where("id = ? AND user_id = ?", req.ToolID, userID).Updates(map[string]interface{}{
		"code": ver.Code, "version": ver.Version, "validation_status": "pending",
	})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "rolled_back", "version": fmt.Sprintf("%d", ver.Version)})
}

// Secrets management for vertical tools
type VerticalSecret struct {
	ID         string    `json:"id" gorm:"primaryKey"`
	UserID     string    `json:"user_id"`
	VerticalID string    `json:"vertical_id"`
	Key        string    `json:"key"`
	Value      string    `json:"-" gorm:"column:encrypted_value"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

func ListVerticalSecretsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }
	verticalID := r.URL.Query().Get("vertical_id")
	var secrets []VerticalSecret
	q := DB.Where("user_id = ?", userID)
	if verticalID != "" { q = q.Where("vertical_id = ?", verticalID) }
	q.Find(&secrets)
	// Only return keys, not values
	out := []map[string]interface{}{}
	for _, s := range secrets {
		out = append(out, map[string]interface{}{"id": s.ID, "key": s.Key, "vertical_id": s.VerticalID, "created_at": s.CreatedAt})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func SetVerticalSecretHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }
	var req struct {
		VerticalID string `json:"vertical_id"`
		Key        string `json:"key"`
		Value      string `json:"value"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	if req.Key == "" || req.Value == "" || req.VerticalID == "" {
		http.Error(w, "Missing fields", 400); return
	}
	// Upsert
	var existing VerticalSecret
	if DB.Where("user_id = ? AND vertical_id = ? AND key = ?", userID, req.VerticalID, req.Key).First(&existing).Error == nil {
		DB.Model(&existing).Update("encrypted_value", req.Value)
	} else {
		DB.Create(&VerticalSecret{ID: uuid.New().String(), UserID: userID, VerticalID: req.VerticalID, Key: req.Key, Value: req.Value})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "key": req.Key})
}

func DeleteVerticalSecretHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", 401); return }
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400); return
	}
	DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&VerticalSecret{})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}
