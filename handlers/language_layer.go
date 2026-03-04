package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"os"

	"github.com/google/uuid"
)

// ─── Function Registry ───
// 6 functions the Language Layer LLM can call

func GetOpenAIFunctionDefinitions() []OpenAITool {
	return []OpenAITool{
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "run_prediction",
				Description: "Run Schema's neural network on a single data row. Returns the model's prediction, confidence score, and class probabilities. Use when the user provides row data and wants to know what Schema predicts.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"row_data": map[string]interface{}{
							"type":        "object",
							"description": "Column name keys with data values",
						},
					},
					"required": []string{"row_data"},
				},
			},
		},
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "run_full_inference",
				Description: "Run the complete vertical AI pipeline on a data row: Schema prediction + all registered tools + agent logic. Returns the full structured response including final_decision. Use when the user wants a complete analysis, not just the model's raw prediction.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"row_data": map[string]interface{}{
							"type":        "object",
							"description": "Column name keys with data values",
						},
					},
					"required": []string{"row_data"},
				},
			},
		},
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "run_tool",
				Description: "Run one specific registered tool on demand. Use when the user asks for a particular calculation or check without running the whole pipeline.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"tool_name": map[string]interface{}{
							"type":        "string",
							"description": "Name of the registered tool",
						},
						"row_data": map[string]interface{}{
							"type":        "object",
							"description": "Column name keys with data values",
						},
						"schema_output": map[string]interface{}{
							"type":        "object",
							"description": "Must include prediction and confidence",
						},
					},
					"required": []string{"tool_name", "row_data", "schema_output"},
				},
			},
		},
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "lookup_prediction",
				Description: "Retrieve a previously completed prediction by its ID. Use when the user references a past decision or asks why something was decided.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"prediction_id": map[string]interface{}{
							"type":        "string",
							"description": "The prediction ID to look up",
						},
					},
					"required": []string{"prediction_id"},
				},
			},
		},
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "query_predictions",
				Description: "Search this vertical's prediction history with filters. Use for summary questions like 'how many fraudulent claims today', 'show me low-confidence predictions from this week'.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"filters": map[string]interface{}{
							"type":        "object",
							"description": "Filter object. Supported keys: prediction (string), last_n (int), date_from (ISO date), date_to (ISO date), confidence_below (float), confidence_above (float), final_decision (string)",
						},
					},
					"required": []string{"filters"},
				},
			},
		},
		{
			Type: "function",
			Function: OpenAIFunction{
				Name:        "get_config",
				Description: "Read this vertical's system configuration. Use when the user asks about thresholds, rules, tool settings, agent behavior, or any configured parameter.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	}
}

func GetClaudeFunctionDefinitions() []ClaudeTool {
	openAITools := GetOpenAIFunctionDefinitions()
	claudeTools := make([]ClaudeTool, len(openAITools))
	for i, t := range openAITools {
		claudeTools[i] = ClaudeTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		}
	}
	return claudeTools
}

// ─── Function Call Bridge ───

type FunctionCallResult struct {
	FunctionName string      `json:"function_name"`
	Arguments    interface{} `json:"arguments"`
	Result       interface{} `json:"result"`
	Error        string      `json:"error,omitempty"`
	ExecutionMs  int64       `json:"execution_ms"`
}

// ExecuteFunctionCall routes a function call from the LLM to the appropriate Runtime operation
func ExecuteFunctionCall(userID, verticalID, modelID, functionName string, arguments json.RawMessage) FunctionCallResult {
	start := time.Now()
	result := FunctionCallResult{
		FunctionName: functionName,
		Arguments:    arguments,
	}

	var err error
	var output interface{}

	switch functionName {
	case "run_prediction":
		output, err = bridgeRunPrediction(userID, modelID, arguments)
	case "run_full_inference":
		output, err = bridgeRunFullInference(userID, modelID, arguments)
	case "run_tool":
		output, err = bridgeRunTool(userID, modelID, arguments)
	case "lookup_prediction":
		output, err = bridgeLookupPrediction(userID, verticalID, arguments)
	case "query_predictions":
		output, err = bridgeQueryPredictions(userID, verticalID, arguments)
	case "get_config":
		output, err = bridgeGetConfig(userID, verticalID)
	default:
		err = fmt.Errorf("unknown function: %s", functionName)
	}

	result.ExecutionMs = time.Since(start).Milliseconds()
	if err != nil {
		result.Error = err.Error()
	} else {
		result.Result = output
	}

	// Audit log
	fmt.Printf("[LANGUAGE_LAYER] Function: %s, User: %s, Vertical: %s, Duration: %dms, Error: %v\n",
		functionName, userID, verticalID, result.ExecutionMs, err)

	// Save to DB
	go saveFunctionCallLog(userID, verticalID, functionName, arguments, result)

	return result
}

func saveFunctionCallLog(userID, verticalID, functionName string, arguments json.RawMessage, result FunctionCallResult) {
	summary := ""
	if result.Error != "" {
		summary = "error: " + result.Error
	} else if resultJSON, err := json.Marshal(result.Result); err == nil {
		s := string(resultJSON)
		if len(s) > 200 {
			s = s[:200] + "..."
		}
		summary = s
	}
	errStr := result.Error
	DB.Exec(`INSERT INTO function_call_logs (user_id, vertical_id, function_name, arguments, result_summary, error, execution_ms) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		userID, verticalID, functionName, string(arguments), summary, errStr, result.ExecutionMs)
}

// ─── Bridge Functions ───

// bridgeRunPrediction calls Flask /predict endpoint
func bridgeRunPrediction(userID, modelID string, args json.RawMessage) (interface{}, error) {
	var params struct {
		RowData map[string]interface{} `json:"row_data"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid arguments: %v", err)
	}

	// Get model_path from DB
	var ftm FineTunedModel
	if err := DB.Where("id = ?", modelID).First(&ftm).Error; err != nil {
		return nil, fmt.Errorf("model not found: %v", err)
	}

	payload := map[string]interface{}{
		"user_id":    userID,
		"model_id":   modelID,
		"model_path": ftm.ModelPath,
		"row_data":   params.RowData,
	}
	return callFlask("/predict_single", payload)
}

// bridgeRunFullInference calls Flask /predict with vertical pipeline
func bridgeRunFullInference(userID, modelID string, args json.RawMessage) (interface{}, error) {
	var params struct {
		RowData map[string]interface{} `json:"row_data"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid arguments: %v", err)
	}

	// Get active vertical config
	var config VerticalConfig
	if err := DB.Where("user_id = ? AND model_id = ? AND enabled = true", userID, modelID).First(&config).Error; err != nil {
		return nil, fmt.Errorf("no active vertical config found")
	}

	// Get model_path from DB
	var ftm FineTunedModel
	if err := DB.Where("id = ?", modelID).First(&ftm).Error; err != nil {
		return nil, fmt.Errorf("model not found: %v", err)
	}

	payload := map[string]interface{}{
		"user_id":            userID,
		"model_id":           modelID,
		"model_path":         ftm.ModelPath,
		"row_data":           params.RowData,
		"vertical_config_id": config.ID,
		"run_pipeline":       true,
	}
	result, err := callFlask("/predict_single", payload)
	if err != nil {
		return nil, err
	}

	// Store in prediction_store
	savePrediction(userID, config.ID, modelID, params.RowData, result)

	return result, nil
}

// bridgeRunTool calls Flask to execute a specific tool
func bridgeRunTool(userID, modelID string, args json.RawMessage) (interface{}, error) {
	var params struct {
		ToolName     string                 `json:"tool_name"`
		RowData      map[string]interface{} `json:"row_data"`
		SchemaOutput map[string]interface{} `json:"schema_output"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid arguments: %v", err)
	}

	// Find the tool
	var tool VerticalTool
	if err := DB.Where("user_id = ? AND model_id = ? AND name = ? AND validation_status = ?",
		userID, modelID, params.ToolName, "passed").First(&tool).Error; err != nil {
		return nil, fmt.Errorf("tool '%s' not found or not validated", params.ToolName)
	}

	payload := map[string]interface{}{
		"user_id":       userID,
		"model_id":      modelID,
		"tool_code":     tool.Code,
		"row_data":      params.RowData,
		"schema_output": params.SchemaOutput,
	}
	return callFlask("/execute_tool_api", payload)
}

// bridgeLookupPrediction queries prediction_store by ID
func bridgeLookupPrediction(userID, verticalID string, args json.RawMessage) (interface{}, error) {
	var params struct {
		PredictionID string `json:"prediction_id"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid arguments: %v", err)
	}

	var pred PredictionStore
	if err := DB.Where("id = ? AND user_id = ?", params.PredictionID, userID).First(&pred).Error; err != nil {
		return nil, fmt.Errorf("prediction '%s' not found", params.PredictionID)
	}

	return pred, nil
}

// bridgeQueryPredictions searches prediction_store with filters
func bridgeQueryPredictions(userID, verticalID string, args json.RawMessage) (interface{}, error) {
	var params struct {
		Filters map[string]interface{} `json:"filters"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return nil, fmt.Errorf("invalid arguments: %v", err)
	}

	query := DB.Where("user_id = ?", userID)
	if verticalID != "" {
		query = query.Where("vertical_id = ?", verticalID)
	}

	// Apply filters
	if pred, ok := params.Filters["prediction"].(string); ok && pred != "" {
		query = query.Where("schema_prediction = ?", pred)
	}
	if decision, ok := params.Filters["final_decision"].(string); ok && decision != "" {
		query = query.Where("final_decision = ?", decision)
	}
	if dateFrom, ok := params.Filters["date_from"].(string); ok && dateFrom != "" {
		query = query.Where("created_at >= ?", dateFrom)
	}
	if dateTo, ok := params.Filters["date_to"].(string); ok && dateTo != "" {
		query = query.Where("created_at <= ?", dateTo)
	}
	if confBelow, ok := params.Filters["confidence_below"].(float64); ok {
		query = query.Where("schema_confidence < ?", confBelow)
	}
	if confAbove, ok := params.Filters["confidence_above"].(float64); ok {
		query = query.Where("schema_confidence > ?", confAbove)
	}

	limit := 50
	if lastN, ok := params.Filters["last_n"].(float64); ok && lastN > 0 {
		limit = int(lastN)
		if limit > 500 {
			limit = 500
		}
	}

	var predictions []PredictionStore
	query.Order("created_at desc").Limit(limit).Find(&predictions)

	return map[string]interface{}{
		"count":       len(predictions),
		"predictions": predictions,
	}, nil
}

// bridgeGetConfig returns the vertical's configuration
func bridgeGetConfig(userID, verticalID string) (interface{}, error) {
	var config VerticalConfig
	var err error
	if verticalID != "" {
		err = DB.Where("id = ? AND user_id = ?", verticalID, userID).First(&config).Error
	} else {
		err = DB.Where("user_id = ? AND enabled = true", userID).First(&config).Error
	}
	if err != nil {
		return nil, fmt.Errorf("no vertical config found")
	}

	var tools []VerticalTool
	DB.Where("vertical_id = ? AND user_id = ?", config.ID, userID).Order("execution_order asc").Find(&tools)

	var agents []VerticalAgent
	DB.Where("vertical_id = ? AND user_id = ?", config.ID, userID).Order("pipeline_order asc").Find(&agents)

	toolNames := make([]string, len(tools))
	for i, t := range tools {
		toolNames[i] = t.Name
	}
	agentNames := make([]string, len(agents))
	for i, a := range agents {
		agentNames[i] = a.Name
	}

	return map[string]interface{}{
		"name":        config.Name,
		"description": config.Description,
		"config_yaml": config.ConfigYAML,
		"enabled":     config.Enabled,
		"version":     config.Version,
		"tools":       toolNames,
		"agents":      agentNames,
		"language_config": config.LanguageConfig,
	}, nil
}

// ─── Helpers ───

func callFlask(endpoint string, payload interface{}) (interface{}, error) {
	reqBody, _ := json.Marshal(payload)
	resp, err := http.Post("http://localhost:6000"+endpoint, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("flask call failed: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("flask error %d: %s", resp.StatusCode, string(body))
	}

	var result interface{}
	json.Unmarshal(body, &result)
	return result, nil
}

func savePrediction(userID, verticalID, modelID string, rowData map[string]interface{}, result interface{}) {
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return
	}

	rowDataJSON, _ := json.Marshal(rowData)
	classProbJSON, _ := json.Marshal(resultMap["class_probabilities"])
	toolOutputsJSON, _ := json.Marshal(resultMap["tool_outputs"])
	agentOutputJSON, _ := json.Marshal(resultMap["agent_output"])
	flagsJSON, _ := json.Marshal(resultMap["flags"])

	confidence := 0.0
	if c, ok := resultMap["schema_confidence"].(float64); ok {
		confidence = c
	}
	prediction := ""
	if p, ok := resultMap["schema_prediction"].(string); ok {
		prediction = p
	}
	finalDecision := ""
	if ao, ok := resultMap["agent_output"].(map[string]interface{}); ok {
		if fd, ok := ao["final_decision"].(string); ok {
			finalDecision = fd
		}
	}

	pred := PredictionStore{
		ID:                uuid.New().String(),
		UserID:            userID,
		VerticalID:        verticalID,
		ModelID:           modelID,
		RowData:           string(rowDataJSON),
		SchemaPrediction:  prediction,
		SchemaConfidence:  confidence,
		ClassProbabilities: string(classProbJSON),
		ToolOutputs:       string(toolOutputsJSON),
		AgentOutput:       string(agentOutputJSON),
		FinalDecision:     finalDecision,
		Flags:             string(flagsJSON),
		CreatedAt:         time.Now(),
	}
	DB.Create(&pred)
}

// ─── Language Layer Chat Integration ───

// IsLanguageLayerActive checks if language layer is enabled for this vertical
func IsLanguageLayerActive(userID, modelID string) (bool, string) {
	var config VerticalConfig
	if err := DB.Where("user_id = ? AND model_id = ? AND enabled = true", userID, modelID).First(&config).Error; err != nil {
		return false, ""
	}
	if config.LanguageConfig == "" {
		return false, ""
	}
	var lc map[string]interface{}
	if err := json.Unmarshal([]byte(config.LanguageConfig), &lc); err != nil {
		return false, ""
	}
	if enabled, ok := lc["enabled"].(bool); ok && enabled {
		return true, config.ID
	}
	return false, ""
}

// CallClaudeWithFunctions calls Claude API with function calling support and loops until text response
func CallClaudeWithFunctions(messages []ClaudeMessage, systemPrompt, model, userID, verticalID, modelID string, w http.ResponseWriter) (string, int, []FunctionCallResult, error) {
	apiKey := GetAPIKeyForProvider(userID, verticalID, &LLMProvider{Type: "anthropic", Model: model})
	if apiKey == "" {
		return "", 0, nil, fmt.Errorf("API key not configured. Please add your Anthropic API key in Settings")
	}

	claudeModel := "claude-sonnet-4-20250514"
	switch model {
	case "claude-opus-4", "claude-4-opus":
		claudeModel = "claude-4-opus-20250514"
	case "claude-haiku", "claude-haiku-4-5":
		claudeModel = "claude-3-5-haiku-20241022"
	case "claude-sonnet-4-5":
		claudeModel = "claude-sonnet-4-20250514"
	}

	tools := GetClaudeFunctionDefinitions()
	var allFunctionCalls []FunctionCallResult
	maxIterations := 5 // safety cap on function call loops

	for i := 0; i < maxIterations; i++ {
		claudeReq := ClaudeRequest{
			Model:     claudeModel,
			MaxTokens: 4096,
			System:    systemPrompt,
			Messages:  messages,
			Tools:     tools,
		}

		reqBody, _ := json.Marshal(claudeReq)
		client := &http.Client{Timeout: 120 * time.Second}
		httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", apiKey)
		httpReq.Header.Set("anthropic-version", "2023-06-01")

		resp, err := client.Do(httpReq)
		if err != nil {
			return "", 0, allFunctionCalls, fmt.Errorf("claude API error: %v", err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return "", 0, allFunctionCalls, fmt.Errorf("claude error %d: %s", resp.StatusCode, string(body))
		}

		var claudeResp ClaudeResponse
		json.Unmarshal(body, &claudeResp)

		// Check for tool_use blocks
		hasToolUse := false
		var textParts []string
		var toolUseBlocks []struct {
			ID    string
			Name  string
			Input json.RawMessage
		}

		for _, block := range claudeResp.Content {
			if block.Type == "text" && block.Text != "" {
				textParts = append(textParts, block.Text)
			}
			if block.Type == "tool_use" {
				hasToolUse = true
				inputJSON, _ := json.Marshal(block.Input)
				toolUseBlocks = append(toolUseBlocks, struct {
					ID    string
					Name  string
					Input json.RawMessage
				}{ID: block.ID, Name: block.Name, Input: inputJSON})
			}
		}

		// If no tool use, return text response
		if !hasToolUse || claudeResp.StopReason == "end_turn" && len(toolUseBlocks) == 0 {
			totalTokens := claudeResp.Usage.InputTokens + claudeResp.Usage.OutputTokens
			finalText := ""
			for _, t := range textParts {
				finalText += t
			}
			return finalText, totalTokens, allFunctionCalls, nil
		}

		// Execute function calls and build tool_result messages
		// First, add the assistant's response (with tool_use blocks) to messages
		assistantContent := make([]ClaudeContentBlock, 0)
		for _, block := range claudeResp.Content {
			cb := ClaudeContentBlock{Type: block.Type}
			if block.Type == "text" {
				cb.Text = block.Text
			} else if block.Type == "tool_use" {
				cb.ID = block.ID
				cb.Name = block.Name
				cb.Input = block.Input
			}
			assistantContent = append(assistantContent, cb)
		}
		messages = append(messages, ClaudeMessage{Role: "assistant", Content: assistantContent})

		// Execute each tool call and build tool_result blocks
		toolResults := make([]ClaudeContentBlock, 0)
		for _, tu := range toolUseBlocks {
			result := ExecuteFunctionCall(userID, verticalID, modelID, tu.Name, tu.Input)
			allFunctionCalls = append(allFunctionCalls, result)

			resultJSON, _ := json.Marshal(result.Result)
			if result.Error != "" {
				resultJSON = []byte(fmt.Sprintf(`{"error": "%s"}`, result.Error))
			}

			toolResults = append(toolResults, ClaudeContentBlock{
				Type:      "tool_result",
				ToolUseID: tu.ID,
				Content:   string(resultJSON),
			})
		}
		messages = append(messages, ClaudeMessage{Role: "user", Content: toolResults})

		fmt.Printf("[LANGUAGE_LAYER] Claude function call loop iteration %d, %d tool calls\n", i+1, len(toolUseBlocks))
	}

	return "Maximum function call iterations reached", 0, allFunctionCalls, nil
}

// CallOpenAIWithFunctions calls OpenAI API with function calling support and loops until text response
func CallOpenAIWithFunctions(messages []ChatMessage, model, userID, verticalID, modelID string, w http.ResponseWriter, provider ...*LLMProvider) (string, int, []FunctionCallResult, error) {
	var p *LLMProvider
	if len(provider) > 0 && provider[0] != nil {
		p = provider[0]
	} else {
		p = &LLMProvider{Type: "openai", Model: model}
	}
	apiKey := GetAPIKeyForProvider(userID, verticalID, p)
	if apiKey == "" && p.Type != "custom" {
		return "", 0, nil, fmt.Errorf("API key not configured. Please add your API key in Settings")
	}
	apiURL := "https://api.openai.com/v1/chat/completions"
	if p.Endpoint != "" {
		apiURL = p.Endpoint
	}

	modelMap := map[string]string{
		"gpt-4o":          "gpt-4o",
		"gpt-4o-mini":     "gpt-4o-mini",
		"gpt-4.5-preview": "gpt-4-turbo-preview",
		"gpt-5":           "gpt-4o",
	}
	openAIModel := modelMap[model]
	if openAIModel == "" {
		openAIModel = "gpt-4o"
	}

	tools := GetOpenAIFunctionDefinitions()
	var allFunctionCalls []FunctionCallResult
	maxIterations := 5

	for i := 0; i < maxIterations; i++ {
		openAIReq := OpenAIRequest{
			Model:       openAIModel,
			Messages:    messages,
			MaxTokens:   4096,
			Temperature: 0.7,
			Tools:       tools,
		}

		reqBody, _ := json.Marshal(openAIReq)
		client := &http.Client{Timeout: 120 * time.Second}
		httpReq, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resp, err := client.Do(httpReq)
		if err != nil {
			return "", 0, allFunctionCalls, fmt.Errorf("openai API error: %v", err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return "", 0, allFunctionCalls, fmt.Errorf("openai error %d: %s", resp.StatusCode, string(body))
		}

		var openAIResp OpenAIResponse
		json.Unmarshal(body, &openAIResp)

		if len(openAIResp.Choices) == 0 {
			return "", 0, allFunctionCalls, fmt.Errorf("no choices in response")
		}

		choice := openAIResp.Choices[0]

		// If no tool calls, return text response
		if len(choice.Message.ToolCalls) == 0 || choice.FinishReason == "stop" {
			return choice.Message.Content, openAIResp.Usage.TotalTokens, allFunctionCalls, nil
		}

		// Add assistant message with tool calls to history
		assistantMsg := ChatMessage{
			Role:    "assistant",
			Content: choice.Message.Content,
		}
		assistantMsg.ToolCalls = choice.Message.ToolCalls
		messages = append(messages, assistantMsg)

		// Execute each tool call
		for _, tc := range choice.Message.ToolCalls {
			result := ExecuteFunctionCall(userID, verticalID, modelID, tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			allFunctionCalls = append(allFunctionCalls, result)

			resultJSON, _ := json.Marshal(result.Result)
			if result.Error != "" {
				resultJSON = []byte(fmt.Sprintf(`{"error": "%s"}`, result.Error))
			}

			// Add tool result to messages with tool_call_id
			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    string(resultJSON),
				ToolCallID: tc.ID,
			})
		}

		fmt.Printf("[LANGUAGE_LAYER] OpenAI function call loop iteration %d, %d tool calls\n", i+1, len(choice.Message.ToolCalls))
	}

	return "Maximum function call iterations reached", 0, allFunctionCalls, nil
}

// ─── LLM Provider Abstraction ───

type LLMProvider struct {
	Type     string `json:"type"`     // openai, anthropic, gemini, ministral, custom
	Model    string `json:"model"`
	APIKeyEnv string `json:"api_key_env"` // secret name from llm_secrets
	Endpoint string `json:"endpoint"`    // custom endpoint URL
}

// GetProviderForVertical returns the configured LLM provider for a vertical
func GetProviderForVertical(userID, verticalID string) *LLMProvider {
	var config VerticalConfig
	if verticalID != "" {
		DB.Where("id = ? AND user_id = ?", verticalID, userID).First(&config)
	} else {
		DB.Where("user_id = ? AND enabled = true", userID).First(&config)
	}

	if config.LanguageConfig == "" {
		return nil
	}

	var lc map[string]interface{}
	if err := json.Unmarshal([]byte(config.LanguageConfig), &lc); err != nil {
		return nil
	}

	providerMap, ok := lc["provider"].(map[string]interface{})
	if !ok {
		return &LLMProvider{Type: "openai", Model: "gpt-4o"} // default
	}

	provider := &LLMProvider{}
	if t, ok := providerMap["type"].(string); ok {
		provider.Type = t
	}
	if m, ok := providerMap["model"].(string); ok {
		provider.Model = m
	}
	if k, ok := providerMap["api_key_env"].(string); ok {
		provider.APIKeyEnv = k
	}
	if e, ok := providerMap["endpoint"].(string); ok {
		provider.Endpoint = e
	}

	return provider
}

// GetAPIKeyForProvider resolves the API key - either from env or from llm_secrets
func GetAPIKeyForProvider(userID, verticalID string, provider *LLMProvider) string {
	// Check user plan
	isUnlimited := false
	var quota UserQuota
if err := DB.Raw("SELECT * FROM user_quotas WHERE user_id = ? LIMIT 1", userID).Scan(&quota).Error; err == nil && quota.ID != "" {
		isUnlimited = quota.Plan == "alpha_unlimited"
fmt.Printf("[LANGUAGE_LAYER] User %s plan=%s isUnlimited=%v\n", userID, quota.Plan, isUnlimited)
} else {
fmt.Printf("[LANGUAGE_LAYER] Quota not found for user %s\n", userID)
}

	// 1. Always check llm_secrets first (user's own key)
	var secret LLMSecret
	secretNames := []string{}
	switch provider.Type {
	case "openai", "custom":
		secretNames = []string{"OPENAI_API_KEY", "openai"}
	case "anthropic":
		secretNames = []string{"ANTHROPIC_API_KEY", "anthropic"}
	case "gemini":
		secretNames = []string{"GEMINI_API_KEY", "gemini"}
case "mistral":
secretNames = []string{"MISTRAL_API_KEY", "mistral"}
	}
	if provider.APIKeyEnv != "" {
		secretNames = append([]string{provider.APIKeyEnv}, secretNames...)
	}
	for _, name := range secretNames {
		if err := DB.Where("user_id = ? AND secret_name = ?", userID, name).First(&secret).Error; err == nil {
			fmt.Printf("[LANGUAGE_LAYER] Using user own API key: %s\n", name)
			return secret.EncryptedValue
		}
	}
// Also try by provider field
if err := DB.Where("user_id = ? AND provider = ?", userID, provider.Type).First(&secret).Error; err == nil {
fmt.Printf("[LANGUAGE_LAYER] Using user own API key by provider: %s\n", provider.Type)
return secret.EncryptedValue
}

	// 2. Unlimited plan -> fallback to env vars
	if isUnlimited {
		switch provider.Type {
		case "openai", "custom":
			return os.Getenv("OPENAI_API_KEY")
		case "anthropic":
			return os.Getenv("ANTHROPIC_API_KEY")
		case "gemini":
			return os.Getenv("GEMINI_API_KEY")
		case "mistral":
			return os.Getenv("MISTRAL_API_KEY")
		default:
			return os.Getenv("OPENAI_API_KEY")
		}
	}

	// 3. Limited plan without own key -> empty (triggers error)
	fmt.Printf("[LANGUAGE_LAYER] No API key for user=%s provider=%s plan=%s\n", userID, provider.Type, quota.Plan)
	return ""
}

// CallLLMWithFunctions is the unified entry point - routes to the correct provider
func CallLLMWithFunctions(history []ChatMessage, systemPrompt, userID, verticalID, modelID string, provider *LLMProvider, w http.ResponseWriter) (string, int, []FunctionCallResult, error) {
	if provider == nil {
		provider = &LLMProvider{Type: "openai", Model: "gpt-4o"}
	}

	fmt.Printf("[LANGUAGE_LAYER] Provider: %s, Model: %s\n", provider.Type, provider.Model)

	switch provider.Type {
	case "anthropic":
		claudeMessages := make([]ClaudeMessage, 0)
		for _, msg := range history {
			if msg.Role != "system" {
				claudeMessages = append(claudeMessages, ClaudeMessage{Role: msg.Role, Content: msg.Content})
			}
		}
		return CallClaudeWithFunctions(claudeMessages, systemPrompt, provider.Model, userID, verticalID, modelID, w)

	case "openai", "custom":
		msgs := []ChatMessage{{Role: "system", Content: systemPrompt}}
		msgs = append(msgs, history...)
		return CallOpenAIWithFunctions(msgs, provider.Model, userID, verticalID, modelID, w, provider)

	case "gemini":
		apiKey := GetAPIKeyForProvider(userID, verticalID, provider)
		return CallGeminiWithFunctions(history, systemPrompt, provider.Model, apiKey, userID, verticalID, modelID, w)

	case "mistral":
		apiKey := os.Getenv("MISTRAL_API_KEY")
		msgs := []ChatMessage{{Role: "system", Content: systemPrompt}}
		msgs = append(msgs, history...)
		return CallMistralWithFunctions(msgs, provider.Model, apiKey, userID, verticalID, modelID, w)

	default:
		return "", 0, nil, fmt.Errorf("unknown provider type: %s", provider.Type)
	}
}

// ─── LLM Secrets CRUD ───

func SaveLLMSecretHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		VerticalID string `json:"vertical_id"`
		Provider   string `json:"provider"`
		SecretName string `json:"secret_name"`
		Value      string `json:"value"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	// Upsert
	var existing LLMSecret
	if err := DB.Where("user_id = ? AND vertical_id = ? AND secret_name = ?",
		userID, req.VerticalID, req.SecretName).First(&existing).Error; err == nil {
		// Update
		existing.EncryptedValue = req.Value // TODO: encrypt
		existing.Provider = req.Provider
		existing.UpdatedAt = time.Now()
		DB.Save(&existing)
	} else {
		// Create
		secret := LLMSecret{
			ID:             uuid.New().String(),
			UserID:         userID,
			VerticalID:     req.VerticalID,
			Provider:       req.Provider,
			SecretName:     req.SecretName,
			EncryptedValue: req.Value, // TODO: encrypt
			CreatedAt:      time.Now(),
			UpdatedAt:      time.Now(),
		}
		DB.Create(&secret)
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func ListLLMSecretsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	verticalID := r.URL.Query().Get("vertical_id")

	var secrets []LLMSecret
	query := DB.Where("user_id = ?", userID)
	if verticalID != "" {
		query = query.Where("vertical_id = ?", verticalID)
	}
	query.Find(&secrets)

	// Mask values
	for i := range secrets {
		if len(secrets[i].EncryptedValue) > 8 {
			secrets[i].EncryptedValue = secrets[i].EncryptedValue[:4] + "****" + secrets[i].EncryptedValue[len(secrets[i].EncryptedValue)-4:]
		} else {
			secrets[i].EncryptedValue = "****"
		}
	}

	json.NewEncoder(w).Encode(secrets)
}

func DeleteLLMSecretHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	var req struct {
		ID string `json:"id"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	DB.Where("id = ? AND user_id = ?", req.ID, userID).Delete(&LLMSecret{})
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// TestLLMConnectionHandler tests if an API key works
func TestLLMConnectionHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	var req struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
		APIKey   string `json:"api_key"`
		Endpoint string `json:"endpoint"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	provider := &LLMProvider{
		Type:     req.Provider,
		Model:    req.Model,
		Endpoint: req.Endpoint,
	}

	// Simple test: send a hello message
	testMsg := []ChatMessage{{Role: "user", Content: "Say hello in one word."}}
	response, _, _, err := CallLLMWithFunctions(testMsg, "You are a test assistant.", userID, "", "", provider, nil)

	if err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "error", "error": err.Error()})
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "ok",
		"response": response,
		"provider": req.Provider,
		"model":    req.Model,
	})
}

// ─── Gemini Provider ───

type GeminiRequest struct {
	Contents         []GeminiContent  `json:"contents"`
	Tools            []GeminiTool     `json:"tools,omitempty"`
	SystemInstruction *GeminiContent  `json:"system_instruction,omitempty"`
	GenerationConfig  map[string]interface{} `json:"generation_config,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text             string                 `json:"text,omitempty"`
	FunctionCall     *GeminiFunctionCall    `json:"functionCall,omitempty"`
	FunctionResponse *GeminiFunctionResponse `json:"functionResponse,omitempty"`
}

type GeminiFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

type GeminiFunctionResponse struct {
	Name     string      `json:"name"`
	Response interface{} `json:"response"`
}

type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"function_declarations"`
}

type GeminiFunctionDeclaration struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

type GeminiResponse struct {
	Candidates []struct {
		Content       GeminiContent `json:"content"`
		FinishReason  string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func GetGeminiFunctionDefinitions() []GeminiTool {
	openAITools := GetOpenAIFunctionDefinitions()
	declarations := make([]GeminiFunctionDeclaration, len(openAITools))
	for i, t := range openAITools {
		declarations[i] = GeminiFunctionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		}
	}
	return []GeminiTool{{FunctionDeclarations: declarations}}
}

// CallMistralWithFunctions calls Mistral API with OpenAI-compatible function calling
func CallMistralWithFunctions(messages []ChatMessage, model, apiKey, userID, verticalID, modelID string, w http.ResponseWriter) (string, int, []FunctionCallResult, error) {
	if apiKey == "" {
		return "", 0, nil, fmt.Errorf("Mistral API key not configured. Add your key in Settings.")
	}

	tools := GetOpenAIFunctionDefinitions()
	totalTokens := 0
	var allFuncCalls []FunctionCallResult
	maxIterations := 5

	for i := 0; i < maxIterations; i++ {
		reqBody := map[string]interface{}{
			"model":    model,
			"messages": messages,
		}
		if len(tools) > 0 && i == 0 {
			reqBody["tools"] = tools
			reqBody["tool_choice"] = "auto"
		}

		jsonBody, _ := json.Marshal(reqBody)
		req, _ := http.NewRequest("POST", "https://api.mistral.ai/v1/chat/completions", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)

		client := &http.Client{Timeout: 120 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return "", 0, nil, fmt.Errorf("Mistral API error: %v", err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return "", 0, nil, fmt.Errorf("Mistral API error (status %d): %s", resp.StatusCode, string(body))
		}

		var result map[string]interface{}
		json.Unmarshal(body, &result)

		if usage, ok := result["usage"].(map[string]interface{}); ok {
			if t, ok := usage["total_tokens"].(float64); ok {
				totalTokens += int(t)
			}
		}

		choices, _ := result["choices"].([]interface{})
		if len(choices) == 0 {
			return "", totalTokens, allFuncCalls, fmt.Errorf("no choices in Mistral response")
		}

		choice := choices[0].(map[string]interface{})
		message := choice["message"].(map[string]interface{})

		toolCalls, hasToolCalls := message["tool_calls"].([]interface{})
		if !hasToolCalls || len(toolCalls) == 0 {
			content, _ := message["content"].(string)
			return content, totalTokens, allFuncCalls, nil
		}

		// Process tool calls - build proper assistant message with tool_calls
		var parsedToolCalls []OpenAIToolCall
		for _, tc := range toolCalls {
			toolCall := tc.(map[string]interface{})
			fn := toolCall["function"].(map[string]interface{})
			fnName, _ := fn["name"].(string)
			fnArgs, _ := fn["arguments"].(string)
			toolCallID, _ := toolCall["id"].(string)
			parsedToolCalls = append(parsedToolCalls, OpenAIToolCall{
				ID:   toolCallID,
				Type: "function",
				Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: fnName, Arguments: fnArgs},
			})
		}
		messages = append(messages, ChatMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: parsedToolCalls,
		})

		for _, tc := range toolCalls {
			toolCall := tc.(map[string]interface{})
			fn := toolCall["function"].(map[string]interface{})
			fnName, _ := fn["name"].(string)
			fnArgs, _ := fn["arguments"].(string)
			toolCallID, _ := toolCall["id"].(string)

			fmt.Printf("[MISTRAL] Tool call: %s(%s)\n", fnName, fnArgs)

			var args map[string]interface{}
			json.Unmarshal([]byte(fnArgs), &args)

			argsJSON, _ := json.Marshal(args)
			fcResult := ExecuteFunctionCall(userID, verticalID, modelID, fnName, argsJSON)
			allFuncCalls = append(allFuncCalls, fcResult)

			resultJSON, _ := json.Marshal(fcResult.Result)
			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    string(resultJSON),
				ToolCallID: toolCallID,
			})
		}
	}

	return "Max tool iterations reached", totalTokens, allFuncCalls, nil
}

func mustJSON(v interface{}) []byte {
	b, _ := json.Marshal(v)
	return b
}

// CallGeminiWithFunctions calls Gemini API with function calling support
func CallGeminiWithFunctions(history []ChatMessage, systemPrompt, model, apiKey, userID, verticalID, modelID string, w http.ResponseWriter) (string, int, []FunctionCallResult, error) {
	if apiKey == "" {
return "", 0, nil, fmt.Errorf("API key not configured. Please add your Gemini API key in Settings")
	}

	if model == "" {
		model = "gemini-2.5-flash"
	}

	tools := GetGeminiFunctionDefinitions()
	var allFunctionCalls []FunctionCallResult
	maxIterations := 5

	// Convert history to Gemini format
	contents := make([]GeminiContent, 0)
	for _, msg := range history {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		if role == "system" {
			continue // system goes to SystemInstruction
		}
		contents = append(contents, GeminiContent{
			Role:  role,
			Parts: []GeminiPart{{Text: msg.Content}},
		})
	}

	for i := 0; i < maxIterations; i++ {
		geminiReq := GeminiRequest{
			Contents: contents,
			Tools:    tools,
			SystemInstruction: &GeminiContent{
				Parts: []GeminiPart{{Text: systemPrompt}},
			},
			GenerationConfig: map[string]interface{}{
				"maxOutputTokens": 4096,
				"temperature":     0.7,
			},
		}

		reqBody, _ := json.Marshal(geminiReq)
		url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", model, apiKey)

		client := &http.Client{Timeout: 120 * time.Second}
		httpReq, _ := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(httpReq)
		if err != nil {
			return "", 0, allFunctionCalls, fmt.Errorf("gemini API error: %v", err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return "", 0, allFunctionCalls, fmt.Errorf("gemini error %d: %s", resp.StatusCode, string(body))
		}

		var geminiResp GeminiResponse
		json.Unmarshal(body, &geminiResp)

		if len(geminiResp.Candidates) == 0 {
			return "", 0, allFunctionCalls, fmt.Errorf("no candidates in gemini response")
		}

		candidate := geminiResp.Candidates[0]
		hasFunctionCall := false
		var textParts []string

		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
			if part.FunctionCall != nil {
				hasFunctionCall = true
			}
		}

		if !hasFunctionCall {
			finalText := ""
			for _, t := range textParts {
				finalText += t
			}
			return finalText, geminiResp.UsageMetadata.TotalTokenCount, allFunctionCalls, nil
		}

		// Add model response to contents
		contents = append(contents, candidate.Content)

		// Execute function calls and add responses
		responseParts := make([]GeminiPart, 0)
		for _, part := range candidate.Content.Parts {
			if part.FunctionCall != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				result := ExecuteFunctionCall(userID, verticalID, modelID, part.FunctionCall.Name, argsJSON)
				allFunctionCalls = append(allFunctionCalls, result)

				responseParts = append(responseParts, GeminiPart{
					FunctionResponse: &GeminiFunctionResponse{
						Name:     part.FunctionCall.Name,
						Response: result.Result,
					},
				})
			}
		}

		contents = append(contents, GeminiContent{
			Role:  "user",
			Parts: responseParts,
		})

		fmt.Printf("[LANGUAGE_LAYER] Gemini function call loop iteration %d\n", i+1)
	}

	return "Maximum function call iterations reached", 0, allFunctionCalls, nil
}

// ─── Dynamic Model Discovery ───

type LLMModelInfo struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Provider string `json:"provider"`
}

// KeyStatusHandler returns which providers have API keys configured
func KeyStatusHandler(w http.ResponseWriter, r *http.Request) {
userID := r.Header.Get("X-User-ID")
if userID == "" {
http.Error(w, "unauthorized", 401)
return
}

// Check plan
isUnlimited := false
var quota UserQuota
if err := DB.Raw("SELECT * FROM user_quotas WHERE user_id = ? LIMIT 1", userID).Scan(&quota).Error; err == nil && quota.ID != "" {
isUnlimited = quota.Plan == "alpha_unlimited"
}

// If unlimited, all providers have keys (from env)
if isUnlimited {
w.Header().Set("Content-Type", "application/json")
json.NewEncoder(w).Encode(map[string]interface{}{
"openai": true, "anthropic": true, "gemini": true, "unlimited": true,
})
return
}

// Check llm_secrets for each provider
providers := []string{"openai", "anthropic", "gemini"}
result := map[string]interface{}{"unlimited": false}
for _, p := range providers {
var count int64
DB.Model(&LLMSecret{}).Where("user_id = ? AND provider = ?", userID, p).Count(&count)
result[p] = count > 0
}
w.Header().Set("Content-Type", "application/json")
json.NewEncoder(w).Encode(result)
}

// ListAvailableModelsHandler returns available LLM models per provider
func ListAvailableModelsHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("[LANGUAGE_LAYER] ListAvailableModels called")
	models := []LLMModelInfo{
		// OpenAI - stable models
		{ID: "gpt-4o", Name: "GPT-4o", Provider: "OpenAI"},
		{ID: "gpt-4o-mini", Name: "GPT-4o Mini", Provider: "OpenAI"},
		// Anthropic - stable models
		{ID: "claude-sonnet-4-5", Name: "Claude Sonnet 4.5", Provider: "Anthropic"},
		{ID: "claude-opus-4", Name: "Claude Opus 4", Provider: "Anthropic"},
		{ID: "claude-haiku-4-5", Name: "Claude Haiku 4.5", Provider: "Anthropic"},
	}

	// Gemini - fetch from API
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey != "" {
		geminiModels := fetchGeminiModels(geminiKey)
		models = append(models, geminiModels...)
	}

	// Mistral - fetch from API, fallback to defaults
	mistralKey := os.Getenv("MISTRAL_API_KEY")
	if mistralKey != "" {
		mistralModels := fetchMistralModels(mistralKey)
		if len(mistralModels) > 0 {
			models = append(models, mistralModels...)
		} else {
			models = append(models, defaultMistralModels()...)
		}
	} else {
		models = append(models, defaultMistralModels()...)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

func defaultMistralModels() []LLMModelInfo {
	return []LLMModelInfo{
		{ID: "ministral-3b-2501", Name: "Schema-3B", Provider: "Schema"},
		{ID: "ministral-8b-2410", Name: "Schema-8B", Provider: "Schema"},
		{ID: "mistral-small-2503", Name: "Schema-14B", Provider: "Schema"},
		{ID: "mistral-medium-2505", Name: "Schema-24B", Provider: "Schema"},
	}
}

func fetchMistralModels(apiKey string) []LLMModelInfo {
	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", "https://api.mistral.ai/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("[MISTRAL] Failed to fetch models: %v\n", err)
		return nil
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		fmt.Printf("[MISTRAL] Models API error %d: %s\n", resp.StatusCode, string(body))
		return nil
	}

	var result struct {
		Data []struct {
			ID           string   `json:"id"`
			Capabilities struct {
				FunctionCalling bool `json:"function_calling"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	json.Unmarshal(body, &result)

	// Name mapping for branding
	nameMap := map[string]string{
		"ministral-3b-2501":    "Schema-3B",
		"ministral-8b-2410":    "Schema-8B",
		"mistral-small-2503":   "Schema-14B",
		"mistral-medium-2505":  "Schema-24B",
		"mistral-small-latest": "Schema-14B",
		"mistral-large-latest": "Schema-Large",
	}

	var models []LLMModelInfo
	for _, m := range result.Data {
		if !m.Capabilities.FunctionCalling {
			continue
		}
		name := m.ID
		if mapped, ok := nameMap[name]; ok {
			name = mapped
		}
		models = append(models, LLMModelInfo{ID: m.ID, Name: name, Provider: "Schema"})
	}
	fmt.Printf("[MISTRAL] Fetched %d models with function calling\n", len(models))
	return models
}

func fetchGeminiModels(apiKey string) []LLMModelInfo {
	// Cached Gemini model list - updated periodically
	// API call yapılmıyor, stabil model listesi döndürülüyor
	return []LLMModelInfo{
		{ID: "gemini-2.5-flash", Name: "Gemini 2.5 Flash", Provider: "Google"},
		{ID: "gemini-2.5-pro", Name: "Gemini 2.5 Pro", Provider: "Google"},
		{ID: "gemini-2.5-flash-lite", Name: "Gemini 2.5 Flash-Lite", Provider: "Google"},
		{ID: "gemini-2.0-flash-001", Name: "Gemini 2.0 Flash 001", Provider: "Google"},
		{ID: "gemini-2.0-flash-lite", Name: "Gemini 2.0 Flash-Lite", Provider: "Google"},
	}
}
