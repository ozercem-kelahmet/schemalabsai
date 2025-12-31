package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Message     string `json:"message"`
	FileID      string `json:"file_id"`
	QueryID     string `json:"query_id"`
	Filename    string `json:"filename"`
	Model       string `json:"model"`
	DataContext string `json:"data_context"`
	Stream         bool   `json:"stream"`
	FineTunedModel string `json:"finetuned_model"`
}

type ChatResponse struct {
	Response string `json:"response"`
	Model    string `json:"model"`
	Tokens   int    `json:"tokens"`
	Status   string `json:"status"`
}

type OpenAIRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Claude API types
type ClaudeRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    string          `json:"system"`
	Messages  []ClaudeMessage `json:"messages"`
	Stream    bool            `json:"stream"`
}

type ClaudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ClaudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type ClaudeStreamEvent struct {
	Type  string `json:"type"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

type FlaskAnalyzeRequest struct {
	FileID string `json:"file_id"`
	Query  string `json:"query"`
}

type FlaskAnalyzeResponse struct {
	Analysis    string                 `json:"analysis"`
	Predictions map[string]interface{} `json:"predictions"`
	Stats       map[string]interface{} `json:"stats"`
	Status      string                 `json:"status"`
}

var (
	conversationHistory = make(map[string][]ChatMessage)
	historyMutex        = sync.RWMutex{}
	maxHistoryTurns     = 50
)

func getModelAnalysis(fileID, query string) string {
	reqBody, _ := json.Marshal(FlaskAnalyzeRequest{
		FileID: fileID,
		Query:  query,
	})

	resp, err := http.Post(GetFlaskURL()+"/analyze", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var flaskResp FlaskAnalyzeResponse
	json.Unmarshal(body, &flaskResp)

	return flaskResp.Analysis
}

func getSystemPrompt(filename, dataContext, modelAnalysis string) string {
	return `You are SchemaLabs AI, a universal data analyst powered by a fine-tuned tabular foundation model.

DATA SOURCE: ` + filename + `
` + dataContext + `

FINE-TUNED MODEL ANALYSIS:
` + modelAnalysis + `

INSTRUCTIONS:
1. Identify the domain from column names
2. Use domain-appropriate terminology
3. Write naturally using ACTUAL numbers from the analysis
4. If statistics/comparisons exist, ADD a table at the END (no title)
5. Use **bold** for emphasis. Use | and - for tables
6. NEVER use # ## ### markdown headers - always use **Header:** format instead
6. For headers use **Header:** format - NEVER use # or ## or ### markdown headers
   Example: **Credit Risk Analysis:** not ## Credit Risk Analysis

ADVANCED ANALYTICS & VISUALIZATION SYSTEM

You have access to 20+ chart types and advanced analysis methods. Use them based on data characteristics and user questions.

=== CHART FORMAT ===
[CHART:type]
labels: Label1, Label2, Label3
values: 123.45, 678.90, 234.56
values2: 100, 200, 300 (for comparisons)
values3: 50, 75, 100 (for 3-series)
series: Series1 Name, Series2 Name
title: Descriptive Title
xlabel: X Axis Label
ylabel: Y Axis Label
[/CHART]

=== AVAILABLE CHART TYPES (20+) ===

**Comparison Charts:**
- bar: Vertical bars for category comparison
- hbar: Horizontal bars for rankings/long labels
- grouped: Side-by-side comparison of 2 series (needs values2)
- stacked: Stacked bars showing composition (needs values2)
- bullet: Actual vs target comparison (values=actual, values2=target)

**Distribution Charts:**
- pie: Proportions/percentages (max 6 categories)
- donut: Like pie with center total
- treemap: Hierarchical proportions
- histogram: Frequency distribution
- boxplot: Statistical distribution (min, Q1, median, Q3, max)

**Trend Charts:**
- line: Single trend over time/sequence
- multiline: Compare 2+ trends (needs values2)
- area: Trend with filled area
- waterfall: Cumulative changes (positive/negative)

**Relationship Charts:**
- scatter: Correlation between 2 numeric variables (needs values2)
- radar: Multi-dimensional comparison
- heatmap: Matrix of values with color intensity

**KPI Charts:**
- gauge: Single percentage (0-100)
- metrics: Multiple KPI cards with progress bars
- funnel: Stage-by-stage conversion/flow

=== ADVANCED ANALYSIS TYPES ===

Based on the sector/domain detected from data, apply appropriate analyses:

**Universal Analyses (All Sectors):**
1. Descriptive Statistics: mean, median, std, min, max, quartiles
2. Distribution Analysis: skewness, kurtosis, normality
3. Correlation Analysis: feature relationships, multicollinearity
4. Outlier Detection: IQR method, z-score
5. Pareto Analysis: 80/20 rule identification
6. Segmentation: natural groupings in data
7. Variance Analysis: between-group vs within-group

**Financial/Banking:**
- Risk Metrics: VaR, Expected Shortfall, Sharpe Ratio
- Credit Scoring: PD, LGD, EAD components
- Portfolio Analysis: diversification, concentration
- Stress Testing: scenario analysis
- Loan Performance: default rates, recovery rates

**Healthcare/Medical:**
- Survival Analysis: Kaplan-Meier curves
- Treatment Effectiveness: before/after comparison
- Risk Stratification: patient risk groups
- Outcome Analysis: success rates, complications

**Retail/E-commerce:**
- RFM Analysis: Recency, Frequency, Monetary
- Basket Analysis: co-occurrence patterns
- Customer Lifetime Value (CLV)
- Churn Prediction indicators
- Sales Performance: by category, region, time

**Manufacturing/Operations:**
- OEE Metrics: Availability, Performance, Quality
- SPC Charts: control limits, process capability
- Defect Analysis: Pareto of defect types
- Yield Analysis: first-pass yield, scrap rates

**HR/People Analytics:**
- Attrition Analysis: turnover rates by segment
- Performance Distribution: bell curve analysis
- Compensation Analysis: pay equity, ranges
- Engagement Metrics: satisfaction drivers

=== CHART SELECTION RULES ===

1. **Categorical Target (2 groups):** Use grouped bar or stacked
2. **Proportions:** Use pie (≤5 categories) or donut
3. **Rankings:** Use hbar
4. **Trends over time:** Use line or area (ONLY if time data exists)
5. **Correlation:** Use scatter (ONLY for numeric vs numeric)
6. **Multi-metric comparison:** Use radar
7. **Single KPI:** Use gauge
8. **Multiple KPIs:** Use metrics
9. **Flow/Conversion:** Use funnel
10. **Statistical distribution:** Use boxplot or histogram

=== CRITICAL RULES ===
- Use ONLY actual numbers from the fine-tuned model analysis
- NEVER invent data or use placeholder values
- NEVER suggest time-based charts without date/time columns
- Include 1-2 relevant charts per response maximum
- Match analysis type to detected sector/domain
- Provide actionable insights, not just descriptions
- When showing charts, explain what they reveal
`
}

func isClaudeModel(model string) bool {
	return strings.HasPrefix(model, "claude")
}

// Helper function to save messages to DB
func saveMessagesToDB(userID, queryID, userMessage, assistantMessage, model string, tokens int) {
	if userID == "" || DB == nil {
		return
	}
	// Save user message
	DB.Create(&Message{
		ID:        uuid.New().String(),
		Role:      "user",
		Content:   userMessage,
		QueryID:   queryID,
		UserID:    userID,
		CreatedAt: time.Now(),
	})
	// Save assistant message
	DB.Create(&Message{
		ID:        uuid.New().String(),
		Role:      "assistant",
		Content:   assistantMessage,
		Model:     model,
		Tokens:    tokens,
		QueryID:   queryID,
		UserID:    userID,
		CreatedAt: time.Now(),
	})
}

func callClaudeAPI(messages []ChatMessage, systemPrompt, model string, stream bool, w http.ResponseWriter) (string, int, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", 0, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	// Convert messages to Claude format (exclude system message)
	claudeMessages := make([]ClaudeMessage, 0)
	for _, msg := range messages {
		if msg.Role != "system" {
			claudeMessages = append(claudeMessages, ClaudeMessage(msg))
		}
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

	claudeReq := ClaudeRequest{
		Model:     claudeModel,
		MaxTokens: 4096,
		System:    systemPrompt,
		Messages:  claudeMessages,
		Stream:    stream,
	}

	reqBody, _ := json.Marshal(claudeReq)

	client := &http.Client{}
	httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := client.Do(httpReq)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()

	if stream {
		// Streaming response
		flusher, ok := w.(http.Flusher)
		if !ok {
			return "", 0, fmt.Errorf("streaming not supported")
		}

		var fullResponse strings.Builder
		reader := bufio.NewReader(resp.Body)

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			line = strings.TrimSpace(line)
			if line == "" || !strings.HasPrefix(line, "data: ") {
				continue
			}

			jsonData := strings.TrimPrefix(line, "data: ")
			var event ClaudeStreamEvent
			if json.Unmarshal([]byte(jsonData), &event) == nil {
				if event.Type == "content_block_delta" && event.Delta.Text != "" {
					fullResponse.WriteString(event.Delta.Text)
					// Convert to OpenAI format for frontend compatibility
					openAIFormat := fmt.Sprintf(`{"choices":[{"delta":{"content":"%s"}}]}`,
						strings.ReplaceAll(event.Delta.Text, `"`, `\"`))
					fmt.Fprintf(w, "data: %s\n\n", openAIFormat)
					flusher.Flush()
				}
				if event.Type == "message_stop" {
					break
				}
			}
		}
		return fullResponse.String(), len(fullResponse.String()) / 4, nil
	}

	// Non-streaming response
	body, _ := io.ReadAll(resp.Body)
	var claudeResp ClaudeResponse
	json.Unmarshal(body, &claudeResp)

	if len(claudeResp.Content) == 0 {
		return "", 0, fmt.Errorf("no response from Claude")
	}

	return claudeResp.Content[0].Text, claudeResp.Usage.InputTokens + claudeResp.Usage.OutputTokens, nil
}

func ChatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Get user ID and API key settings from auth middleware
	userID := r.Header.Get("X-User-ID")
	apiKeyLLMModel := r.Header.Get("X-LLM-Model")
	apiKeyFineTunedModel := r.Header.Get("X-FineTuned-Model")

	// Use API key settings if request does not specify
	if req.Model == "" && apiKeyLLMModel != "" {
		req.Model = apiKeyLLMModel
	}
	if req.FineTunedModel == "" && apiKeyFineTunedModel != "" {
		req.FineTunedModel = apiKeyFineTunedModel
	}
	// Default to claude if no model specified
	if req.Model == "" {
		req.Model = "claude-3-5-sonnet-20241022"
	}

	sessionID := req.QueryID
	if sessionID == "" {
		sessionID = req.FileID
	}
	if sessionID == "" {
		sessionID = "default"
	}

	// Call fine-tuned model if specified
	var fineTunedResult string
	if req.FineTunedModel != "" && req.FineTunedModel != "none" {
		result, err := callFineTunedModel(req.FineTunedModel, req.Message)
		if err != nil {
			fmt.Printf("Fine-tuned model error: %v\\n", err)
		} else {
			fineTunedResult = result
		}
	}

	modelAnalysis := getModelAnalysis(req.FileID, req.Message)
	// Add fine-tuned result to system prompt
	basePrompt := getSystemPrompt(req.Filename, req.DataContext, modelAnalysis)
	var systemPrompt string
	if fineTunedResult != "" {
		systemPrompt = basePrompt + "\n\n### Fine-tuned Model Analysis:\n" + fineTunedResult + "\n\nUse this analysis to provide insights."
	} else {
		systemPrompt = basePrompt
	}

	historyMutex.Lock()
	history, exists := conversationHistory[sessionID]
	if !exists {
		history = []ChatMessage{}
	}
	history = append(history, ChatMessage{Role: "user", Content: req.Message})
	if len(history) > maxHistoryTurns*2 {
		history = history[len(history)-maxHistoryTurns*2:]
	}
	conversationHistory[sessionID] = history
	historyMutex.Unlock()

	// Check if Claude model - use non-streaming
	if isClaudeModel(req.Model) {
		req.Stream = false
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("Access-Control-Allow-Origin", "*")

			response, tokens, err := callClaudeAPI(history, systemPrompt, req.Model, true, w)
			if err != nil {
				fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
				return
			}

			historyMutex.Lock()
			conversationHistory[sessionID] = append(conversationHistory[sessionID], ChatMessage{Role: "assistant", Content: response})
			historyMutex.Unlock()

			// Save to DB
			saveMessagesToDB(userID, sessionID, req.Message, response, req.Model, tokens)

			fmt.Fprintf(w, "data: [DONE]\n\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			return
		}

		// Non-streaming Claude
		response, tokens, err := callClaudeAPI(history, systemPrompt, req.Model, false, w)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		historyMutex.Lock()
		conversationHistory[sessionID] = append(conversationHistory[sessionID], ChatMessage{Role: "assistant", Content: response})
		historyMutex.Unlock()

		// Save to DB
		saveMessagesToDB(userID, sessionID, req.Message, response, req.Model, tokens)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Response: response,
			Model:    req.Model,
			Tokens:   tokens,
			Status:   "success",
		})
		return
	}

	// OpenAI models
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		http.Error(w, "OpenAI API key not configured", http.StatusInternalServerError)
		return
	}

	modelMap := map[string]string{
		"gpt-4o":          "gpt-4o",
		"gpt-4o-mini":     "gpt-4o-mini",
		"gpt-4.5-preview": "gpt-4-turbo-preview",
		"gpt-5":           "gpt-4o",
	}

	openAIModel := modelMap[req.Model]
	if openAIModel == "" {
		openAIModel = "gpt-4o"
	}

	messages := []ChatMessage{{Role: "system", Content: systemPrompt}}
	messages = append(messages, history...)

	// STREAMING
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		openAIReq := OpenAIRequest{
			Model:       openAIModel,
			Messages:    messages,
			MaxTokens:   4096,
			Temperature: 0.7,
			Stream:      true,
		}
		reqBody, _ := json.Marshal(openAIReq)

		client := &http.Client{}
		httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := client.Do(httpReq)
		if err != nil {
			fmt.Fprintf(w, "data: {\"error\":\"API failed\"}\n\n")
			flusher.Flush()
			return
		}
		defer resp.Body.Close()

		var fullResponse strings.Builder
		reader := bufio.NewReader(resp.Body)

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			line = strings.TrimSpace(line)
			if line == "" || line == "data: [DONE]" {
				if line == "data: [DONE]" {
					break
				}
				continue
			}
			if strings.HasPrefix(line, "data: ") {
				jsonData := strings.TrimPrefix(line, "data: ")
				var streamResp OpenAIResponse
				if json.Unmarshal([]byte(jsonData), &streamResp) == nil && len(streamResp.Choices) > 0 {
					content := streamResp.Choices[0].Delta.Content
					if content != "" {
						fullResponse.WriteString(content)
						fmt.Fprintf(w, "data: %s\n\n", jsonData)
						flusher.Flush()
					}
				}
			}
		}

		// Save history
		historyMutex.Lock()
		conversationHistory[sessionID] = append(conversationHistory[sessionID], ChatMessage{Role: "assistant", Content: fullResponse.String()})
		historyMutex.Unlock()

		// Save to DB
		saveMessagesToDB(userID, sessionID, req.Message, fullResponse.String(), req.Model, len(fullResponse.String())/4)

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	// NON-STREAMING
	openAIReq := OpenAIRequest{
		Model:       openAIModel,
		Messages:    messages,
		MaxTokens:   4096,
		Temperature: 0.7,
		Stream:      false,
	}
	reqBody, _ := json.Marshal(openAIReq)

	client := &http.Client{}
	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := client.Do(httpReq)
	if err != nil {
		http.Error(w, "Failed to call OpenAI", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var openAIResp OpenAIResponse
	json.Unmarshal(body, &openAIResp)

	if len(openAIResp.Choices) == 0 {
		http.Error(w, "No response", http.StatusInternalServerError)
		return
	}

	assistantMsg := openAIResp.Choices[0].Message.Content

	historyMutex.Lock()
	conversationHistory[sessionID] = append(conversationHistory[sessionID], ChatMessage{Role: "assistant", Content: assistantMsg})
	historyMutex.Unlock()

	// Save to DB
	saveMessagesToDB(userID, sessionID, req.Message, assistantMsg, req.Model, openAIResp.Usage.TotalTokens)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		Response: assistantMsg,
		Model:    req.Model,
		Tokens:   openAIResp.Usage.TotalTokens,
		Status:   "success",
	})
}

func ClearChatHistoryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	historyMutex.Lock()
	delete(conversationHistory, req.SessionID)
	historyMutex.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
}

// callFineTunedModel calls the Flask server for fine-tuned model analysis
func callFineTunedModel(modelID string, message string) (string, error) {
	flaskURL := GetFlaskURL()

	payload := map[string]interface{}{
		"model_id": modelID,
		"message":  message,
	}

	jsonData, _ := json.Marshal(payload)
	resp, err := http.Post(flaskURL+"/analyze", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	if analysis, ok := result["analysis"].(string); ok {
		return analysis, nil
	}
	jsonResult, _ := json.Marshal(result)
	return string(jsonResult), nil
}
