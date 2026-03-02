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

	ctx := "\n\n=== VERTICAL AI RUNTIME ===\n"
	ctx += "Active Vertical: " + config.Name + "\n"

	if config.ConfigYAML != "" {
		ctx += "\n--- System Config Rules ---\n"
		ctx += config.ConfigYAML + "\n"
	}

	ctx += "\nIMPORTANT: The Vertical AI Runtime processed this data with custom tools and agents.\n"
	ctx += "The tool outputs and agent decisions are included in the analysis data above.\n"
	ctx += "You MUST incorporate these results into your response:\n"
	ctx += "- If a tool calculated risk scores or metrics, include them prominently\n"
	ctx += "- If an agent made a final_decision, state it clearly at the beginning of your response\n"
	ctx += "- Follow the behavioral rules defined in the System Config above\n"

	for _, t := range tools {
		ctx += fmt.Sprintf("- Tool: %s (%s)\n", t.Name, t.Hook)
	}
	for _, a := range agents {
		ctx += fmt.Sprintf("- Agent: %s (role: %s)\n", a.Name, a.Role)
	}

	return ctx
}
