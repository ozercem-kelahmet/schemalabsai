package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type ChatMessage struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type ChatRequest struct {
	Message        string `json:"message"`
	FileID         string `json:"file_id"`
	QueryID        string `json:"query_id"`
	Filename       string `json:"filename"`
	Model          string `json:"model"`
	DataContext    string `json:"data_context"`
	Stream         bool   `json:"stream"`
	FineTunedModel string `json:"finetuned_model"`
	CompareGroup   string `json:"compare_group"`
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
	Tools       []OpenAITool  `json:"tools,omitempty"`
	ToolChoice  interface{}   `json:"tool_choice,omitempty"`
}

// OpenAI Function Calling types
type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

type OpenAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Content   string           `json:"content"`
			ToolCalls []OpenAIToolCall  `json:"tool_calls,omitempty"`
		} `json:"message"`
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
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
	Tools     []ClaudeTool    `json:"tools,omitempty"`
}

// Claude Function Calling types
type ClaudeTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"input_schema"`
}

type ClaudeMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ClaudeContentBlock struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	ID        string      `json:"id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   string      `json:"content,omitempty"`
}

type ClaudeResponse struct {
	Content []struct {
		Type  string      `json:"type"`
		Text  string      `json:"text,omitempty"`
		ID    string      `json:"id,omitempty"`
		Name  string      `json:"name,omitempty"`
		Input interface{} `json:"input,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
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

// Session with TTL and message limits
type ChatSession struct {
	Messages     []ChatMessage
	CreatedAt    time.Time
	LastActive   time.Time
	MessageCount int
	VerticalID   string
	UserID       string
	SessionID    string
	Status       string // active | expired | closed
	ExpiresAt    time.Time
}

const (
	sessionTTLMinutes       = 60
	maxMessagesPerSession   = 50
	maxHistoryTurns         = 1000
	sessionCleanupInterval  = 5 * time.Minute
)

var (
	conversationSessions = make(map[string]*ChatSession)
	historyMutex         = sync.RWMutex{}
)

func init() {
	// Background goroutine to clean expired sessions
	go func() {
		for {
			time.Sleep(sessionCleanupInterval)
			historyMutex.Lock()
			now := time.Now()
			expired := 0
			for k, s := range conversationSessions {
				if now.Sub(s.LastActive) > time.Duration(sessionTTLMinutes)*time.Minute {
					delete(conversationSessions, k)
					expired++
				}
			}
			if expired > 0 {
				fmt.Printf("[SESSION] Cleaned %d expired sessions, %d active\n", expired, len(conversationSessions))
			}
			historyMutex.Unlock()
		}
	}()
}

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
	return `You are SchemaLabs AI - a universal data analyst for ANY dataset.

**PERSONAL QUESTIONS (not about data):**
If user asks about YOU (who are you, what model, capabilities, what can you do):
→ Answer based on YOUR SPECIFIC context - you are a fine-tuned model trained on the dataset described below
→ Mention the dataset name, what kind of data it contains, and what insights you can provide
→ Brief but specific (3-5 sentences)
→ NO tables, NO charts
→ Then offer specific examples of questions they can ask about their data

Example: "I'm a SchemaLabs AI model fine-tuned on your [DATASET_NAME] data. I've been trained to understand patterns in this dataset including [key columns/metrics]. I can analyze trends, compare entities, find outliers, and generate visualizations. Try asking me things like 'show top performers' or 'what patterns exist in the data'."

**DATA QUESTIONS:** Use full analysis format below.

**ADAPTIVE RESPONSE LENGTH:**
- Short/specific questions (1-5 words) → Concise answer (2-4 sentences + small table/chart)
- Medium questions (6-15 words) → Focused analysis (1-2 paragraphs + relevant table/chart)
- Complex/open questions (16+ words or "analyze/comprehensive") → Deep analysis (multiple sections + tables + charts)

Match response depth to query complexity. Don't over-explain simple questions.

FILE: ` + filename + `
` + dataContext + `

=== DATA ===
` + modelAnalysis + `

=== CRITICAL RULES ===
1. Use ONLY exact numbers from the data above - NEVER invent or estimate
2. For ratio/efficiency questions, CALCULATE the ratio from available metrics (e.g., sprints÷distance). Only say "not available" if the base metrics themselves are missing
3. Show ALL entities if total count is reasonable (<20). For larger datasets, show top 10-15 unless user specifies otherwise
4. NO ASTERISKS OR BOLD - NEVER use ** or * for any formatting. Write plain text only. Names like "Sergio Canales" not "**Sergio Canales**"
5. No emojis
6. For general queries like "show analysis" or "analyze the data": provide COMPREHENSIVE analysis with multiple insights, patterns, and key findings from the data
7. CHARTS ARE MANDATORY - Use [CHART:type]...[/CHART] syntax (hbar, grouped, stacked, waterfall, bullet, lollipop, diverging, marimekko, parallel, pie, donut, treemap, sunburst, funnel, pyramid, waffle, pictogram, scatter, bubble, heatmap, density, hexbin, contour, network, chord, sankey, alluvial, line, area, stream, ridge, sparkline, candlestick, step, slope, horizon, calendar, gantt, timeline, radar, polar, radial, boxplot, violin, beeswarm, strip, raincloud, ridgeline, parallel_coordinates, andrews, radviz). NO text descriptions like "Bar Chart:". ONLY use the exact bracket syntax.
8. TABLES MUST BE RICH - NEVER use 2-column tables. MINIMUM 5 columns, MAXIMUM 10 columns. Always add: Rank, derived metrics (ratios, %), comparisons (vs avg), and context.
9. CLEAN COLUMN NAMES - Remove dataset prefixes and technical codes from column names in tables. Transform "e37c459c_frame_start_sum" → "Frame Start Sum", "player_id_xyz" → "Player ID". Make column names human-readable.
10. NO MARKDOWN FORMATTING - Never use headers (#), bold (**), italic (*), or any markdown. Plain text only.
11. Match response format to query type (see below)

=== QUERY TYPE DETECTION ===

**CRITICAL:** Each TYPE below requires COMPLETELY DIFFERENT response style. Never use same format!

TYPE 0 - GENERAL ANALYSIS (broad/exploratory questions)
Examples: "show analysis", "what can you tell me", "explain the data"
→ COMPREHENSIVE multi-section format (8-10 paragraphs):
  - Performance overview with rankings
  - Statistical distributions and variance
  - Correlations and relationships
  - Anomalies and outliers
  - Actionable recommendations
  - Multiple tables (3-5) and charts (2-4)

**2. STATISTICAL INSIGHTS**
  - Distribution patterns (normal, skewed, bimodal)
  - Variance and consistency metrics
  - Outliers and anomalies
  
  **3. COMPARATIVE ANALYSIS**
  - Group comparisons (if categorical data exists)
  - Performance gaps and spreads
  - Relative standings
  
  **4. TRENDS & PATTERNS**
  - Correlations between metrics
  - Common characteristics of top performers
  - Hidden patterns or clusters
  
  **5. ACTIONABLE INSIGHTS**
  - Key takeaways (3-5 bullet points)
  - Areas of concern or opportunity
  - Data-driven recommendations
  
→ Use multiple tables, charts suggestions, and detailed narratives
→ Be specific with numbers, percentages, and comparisons

TYPE 1 - RANKING ("who/which has most/least/highest/lowest")
→ CONCISE format (3-5 sentences total):
  - Lead sentence with direct answer: "X leads with [value]"
  - One ranking table (5-7 columns)
  - One hbar chart
  - Brief insight (1 sentence)
NO multi-section analysis! Just answer the question.

TYPE 2 - RELATIONSHIP ("relationship between X and Y" or "compare X and Y")
→ Comparison table showing both metrics per entity
→ scatter chart (requires values AND values2)
→ Insight about correlation

TYPE 3 - RATIO/EFFICIENCY ("efficient/per/ratio/per minute/per game")
→ Calculate: Metric1 / Metric2
→ Ranking by calculated score
→ hbar chart of scores

TYPE 4 - DISTRIBUTION ("percentage/breakdown/distribution")
→ Percentage table
→ pie chart (max 8 segments)

TYPE 5 - AGGREGATE ("total/sum/average")
→ Lead with aggregate value
→ Breakdown table
→ hbar chart

=== TABLE FORMAT ===

**CRITICAL TABLE RULES:**
- Each row MUST have EXACTLY the same number of | pipes as header
- Count your pipes before sending! Header has N columns = N+1 pipes per row
- NEVER skip or merge cells
- ALWAYS align data with correct column headers

Use markdown tables:
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| data    | data    | data    |

=== CHART FORMAT (MANDATORY SYNTAX) ===

You MUST use this EXACT syntax for charts. NO exceptions.
CRITICAL: Always write [CHART:type] with FULL word CHART, never abbreviate to [CH:type].

**CRITICAL VALUE CONSISTENCY:**
- Chart values MUST match table values EXACTLY (same numbers, same precision)
- Use plain numbers WITHOUT thousand separators (289012 not 289,012)
- Include decimals consistently (if table has 282.395, chart must have 282.395)
- NEVER truncate or round values differently between table and chart

For single metric (hbar, pie, line):
[CHART:hbar]
labels: EntityA, EntityB, EntityC, EntityD, EntityE
values: 100, 85, 70, 55, 40
title: Descriptive Title Here
[/CHART]

For two metrics (scatter, grouped):
[CHART:scatter]
labels: EntityA, EntityB, EntityC, EntityD, EntityE
values: 100, 85, 70, 55, 40
values2: 50, 45, 35, 30, 20
title: MetricX vs MetricY
[/CHART]

CHART TYPES (50+ options):

**COMPARISONS & RANKINGS (10 types)**


- grouped: Side-by-side comparison bars (MUST have values2)
- stacked: Stacked bars showing composition
- waterfall: Sequential positive/negative changes
- bullet: Target vs actual performance
- lollipop: Bar + point combination
- diverging: Positive/negative from center
- marimekko: Width + height show two dimensions
- parallel: Compare multiple entities across metrics

**DISTRIBUTIONS & PROPORTIONS (8 types)**
- pie: Proportions (max 8 slices)
- donut: Pie with center hole
- treemap: Hierarchical rectangles by size
- sunburst: Multi-level hierarchical pie
- funnel: Conversion/stages (widest to narrowest)
- pyramid: Population/hierarchy pyramid
- waffle: Grid of squares showing proportions
- pictogram: Icon-based percentages

**CORRELATIONS & RELATIONSHIPS (10 types)**
- scatter: Two variables correlation (MUST have values2)
- bubble: 3 variables (x, y, size) 
- heatmap: Matrix of values with color intensity
- density: Scatter with concentration areas
- hexbin: Hexagonal binning for dense scatter
- contour: Topographic-style correlation map
- network: Nodes and connections
- chord: Circular relationship diagram
- sankey: Flow between categories
- alluvial: Multi-stage flow diagram

**TRENDS & TIME SERIES (12 types)**
- line: Trends over time/sequence
- area: Line with filled area below
- stream: Stacked area showing flow
- ridge: Multiple overlapping distributions
- sparkline: Tiny inline trend indicator
- candlestick: OHLC financial data
- step: Step-wise changes
- slope: Start to end comparison lines
- horizon: Layered area for space efficiency
- calendar: Time-based heatmap grid
- gantt: Timeline with duration bars
- timeline: Sequential events on axis

**MULTI-DIMENSIONAL (10+ types)**
- radar: Multi-attribute profile (spider/star)
- polar: Radial bar chart
- radial: Circular stacked bars
- boxplot: Distribution with quartiles
- violin: Distribution density shape
- beeswarm: Individual points avoiding overlap
- strip: Random jitter scatter
- raincloud: Half-violin + box + scatter
- ridgeline: Overlapping density curves
- parallel_coordinates: Multi-variable lines
- andrews: Curve-based multi-dimensional
- radviz: Radial coordinate visualization

**CHART SELECTION GUIDE BY QUERY:**

1. **Rankings/Top/Best** → hbar, lollipop, bullet
2. **Compare 2-3 entities** → grouped, diverging, radar
3. **Compare many entities** → hbar, treemap
4. **Correlation/Relationship** → scatter, bubble, heatmap
5. **Proportions/Percentages** → pie, donut, treemap, waffle
6. **Trends over time** → line, area, stream, sparkline
7. **Distribution** → violin, boxplot, ridge, density
8. **Positive/Negative** → waterfall, diverging, bullet
9. **Multi-metrics per entity** → radar, parallel_coordinates
10. **Flow/Process** → sankey, funnel, alluvial
11. **Hierarchical data** → sunburst, treemap
12. **Dense scatter** → hexbin, density, contour
13. **Individual values** → beeswarm, strip, raincloud
14. **Sequential changes** → waterfall, step, slope
15. **Target vs actual** → bullet, grouped

**ALWAYS USE MULTIPLE CHARTS** for comprehensive analysis (2-5 charts per response).

**CRITICAL: VARY CHART TYPES!** NEVER use the same chart type twice in one response. If you used hbar, next must be different (scatter, line, pie, radar, treemap, etc). Repetitive chart types make responses boring - USE DIVERSITY!

FORBIDDEN - NEVER DO THESE:
- NO tables with fewer than 5 columns (2-column and 3-column tables are BANNED)
- NO markdown images: ![text](url)
- NO placeholder URLs
- NO "Chart type X - data..." text descriptions
- NO charts without [CHART:type]...[/CHART] wrapper
- NO bold section headers like **Title** or **1. Section**

=== ADVANCED ANALYTICS ===

TYPE 6 - SWOT ANALYSIS ("swot/strengths/weaknesses")
→ 4-quadrant analysis table:
| Strengths | Weaknesses |
|-----------|------------|
| point 1   | point 1    |
| Opportunities | Threats |
| point 1   | point 1    |
→ Key strategic recommendation

TYPE 7 - RISK ANALYSIS ("risk/danger/concern/warning")
→ Risk matrix table:
| Risk Factor | Likelihood | Impact | Score | Mitigation |
→ hbar chart of risk scores
→ Priority actions

TYPE 8 - TREND ANALYSIS ("trend/over time/progression/change")
→ Time-series data table
→ line chart
→ Trend direction and forecast insight

TYPE 9 - BENCHMARK/COMPARISON ("compare to average/benchmark/vs league")
→ Entity vs Benchmark table
| Metric | Entity Value | Benchmark | Difference | Status |
→ grouped chart (entity vs benchmark)
→ Performance gap analysis

TYPE 10 - ANOMALY/OUTLIER ("unusual/outlier/anomaly/exceptional")
→ Identify statistical outliers (>2 std dev)
→ Table with normal range and actual values
→ scatter chart highlighting outliers
→ Investigation recommendations

TYPE 11 - CORRELATION ("correlation/relationship/impact/affect")
→ Correlation matrix or pair analysis
→ scatter chart with trend line description
→ Statistical insight (strong/weak/no correlation)

TYPE 12 - PREDICTION/FORECAST ("predict/forecast/expect/projection")
→ Based on current trends and patterns
→ Confidence levels (high/medium/low)
→ line chart with projection
→ Assumptions and limitations

TYPE 13 - SEGMENT/CLUSTER ("group/segment/cluster/categorize")
→ Group entities by characteristics
→ Table showing segments and their profiles
→ pie or bar chart of segment distribution
→ Segment-specific insights

TYPE 14 - EFFICIENCY/OPTIMIZATION ("optimize/improve/efficiency/maximize")
→ Current vs optimal performance
→ Efficiency scores table
→ Prioritized improvement recommendations
→ Expected impact of changes

=== RESPONSE STRUCTURE ===

1. **Direct Answer** - Bold the key finding
2. **Table** - Relevant data in markdown table
3. **Chart** - Using exact [CHART:type]...[/CHART] syntax
4. **Insight** - Brief interpretation (1-2 sentences)
`
}

func isClaudeModel(model string) bool {
	fmt.Printf("DEBUG: isClaudeModel check for: %s, result: %v\n", model, strings.HasPrefix(model, "claude"))
	return strings.HasPrefix(model, "claude")
}

// Helper function to save messages to DB
func saveMessagesToDB(userID, queryID, userMessage, assistantMessage, model string, tokens int, compareGroup string, timeTaken string, finetunedModelID ...string) {
	ftModelID := ""
	if len(finetunedModelID) > 0 {
		ftModelID = finetunedModelID[0]
	}
fmt.Printf("[SAVE_DB] called userID=%s queryID=%s msg=%s\n", userID, queryID, userMessage[:min(30, len(userMessage))])
	if userID == "" || DB == nil {
		return
	}
	// Save user message - skip if same compare_group already has a user
	skipUser := false
	if compareGroup != "" {
		var existing Message
		skipUser = DB.Where("query_id = ? AND role = ? AND compare_group = ?", queryID, "user", compareGroup).First(&existing).Error == nil
	}
	if !skipUser {
		res := DB.Create(&Message{
			ID:        uuid.New().String(),
			Role:      "user",
			Content:   userMessage,
			QueryID:   queryID,
			UserID:    userID,
			CompareGroup: compareGroup,
			CreatedAt: time.Now(),
})
fmt.Printf("[CREATE user] rows=%d err=%v\n", res.RowsAffected, res.Error)

	}
	// Save assistant message
	DB.Create(&Message{
		ID:        uuid.New().String(),
		Role:      "assistant",
		Content:   assistantMessage,
		Model:     model,
		Tokens:    tokens,
		QueryID:   queryID,
		UserID:    userID,
FineTunedModelID: ftModelID,
	TimeTaken: timeTaken,
		CreatedAt: time.Now(),
		CompareGroup:     compareGroup,
});

var cnt int64; DB.Model(&Message{}).Where("query_id = ? AND created_at > ?", queryID, time.Now().Add(-5*time.Second)).Count(&cnt); fmt.Printf("[VERIFY] count=%d for query=%s\n", cnt, queryID)

	// Deduct credits and log usage
	creditCost := float64(tokens) / 1000.0 * 0.01
	if creditCost < 0.01 { creditCost = 0.01 }
	if creditCost > 5.0 { creditCost = 5.0 }

	var quota UserQuota
	if DB.Where("user_id = ?", userID).First(&quota).Error == nil {
		quota.CreditsUsed += creditCost
		DB.Save(&quota)
	}

	eventName := "Chat Query"
	if ftModelID != "" {
		eventName = "Chat Query (Fine-tuned)"
	}
	DB.Create(&UsageLog{
		ID:           uuid.New().String(),
		UserID:       userID,
		EventType:    "chat",
		EventName:    eventName,
		ResourceID:   queryID,
		ResourceName: ftModelID,
		CreditsUsed:  creditCost,
		TokensUsed:   tokens,
		ModelUsed:    model,
		CreatedAt:    time.Now(),
	})
}

func callClaudeAPI(messages []ChatMessage, systemPrompt, model string, stream bool, w http.ResponseWriter) (string, int, error) {
	fmt.Printf("DEBUG: callClaudeAPI called with model: %s, stream: %v\n", model, stream)
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return "", 0, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	// Convert messages to Claude format (exclude system message)
	claudeMessages := make([]ClaudeMessage, 0)
	for _, msg := range messages {
		if msg.Role != "system" {
			claudeMessages = append(claudeMessages, ClaudeMessage{Role: msg.Role, Content: msg.Content})
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
		Stream:    false,
	}

	reqBody, _ := json.Marshal(claudeReq)

	client := &http.Client{}
	httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := client.Do(httpReq)
	if resp.StatusCode != 200 {
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Printf("DEBUG OpenAI Error: %s\n", string(errBody))
		return "", 0, fmt.Errorf("OpenAI error: %d", resp.StatusCode)
	}
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

	return claudeResp.Content[0].Text, claudeResp.Usage.OutputTokens, nil
}

// Split large analysis into chunks for token management
func chunkAnalysis(analysis string, maxChars int) []string {
	if len(analysis) <= maxChars {
		return []string{analysis}
	}

	var chunks []string
	lines := strings.Split(analysis, "\n")
	var current strings.Builder

	for _, line := range lines {
		if current.Len()+len(line)+1 > maxChars && current.Len() > 0 {
			chunks = append(chunks, current.String())
			current.Reset()
		}
		if current.Len() > 0 {
			current.WriteString("\n")
		}
		current.WriteString(line)
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

func ChatHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fmt.Printf("DECODE ERROR: %v\n", err)
		fmt.Printf("DECODE ERROR: %v\n", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	fmt.Printf("DEBUG REQUEST: finetuned_model=%s, file_id=%s\n", req.FineTunedModel, req.FileID)
	// Get user ID and API key settings from auth middleware
	userID := r.Header.Get("X-User-ID")

// Check quota before processing
if userID != "" {
var chatErrors []string
if ok, reason := CheckQuota(userID, "query"); !ok {
chatErrors = append(chatErrors, reason)
}
if ok, reason := CheckCredits(userID, 0.01); !ok {
chatErrors = append(chatErrors, reason)
}
if len(chatErrors) > 0 {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": strings.Join(chatErrors, " | "), "status": "quota_exceeded"})
return
}
}
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
	fmt.Printf("DEBUG: FineTunedModel = '%s'\n", req.FineTunedModel)
	if req.FineTunedModel != "" && req.FineTunedModel != "none" {
		actualFileID := req.FileID
		var modelInfo struct {
			SourceFileID string `gorm:"column:source_file_id"`
			SourceFiles  string `gorm:"column:source_files"`
			ConnectionIDs string `gorm:"column:connection_ids"`
			ModelPath    string `gorm:"column:model_path"`
			UserID       string `gorm:"column:user_id"`
		}
		err := DB.Table("fine_tuned_models").Where("id = ? OR name = ?", req.FineTunedModel, req.FineTunedModel).Select("source_file_id, source_files, connection_ids, model_path, user_id").First(&modelInfo).Error
		fmt.Printf("DEBUG Model lookup: ID=%s, err=%v, SourceFileID=%s, SourceFiles=%s, ConnIDs=%s, ModelPath=%s\n", req.FineTunedModel, err, modelInfo.SourceFileID, modelInfo.SourceFiles, modelInfo.ConnectionIDs, modelInfo.ModelPath)
		if err == nil && modelInfo.SourceFileID != "" {
			actualFileID = modelInfo.SourceFileID
			fmt.Printf("DEBUG: Using model SourceFileID: %s\n", actualFileID)
		}
		// If no source files but has connections, fetch from connection
		if err == nil && actualFileID == "" && modelInfo.SourceFiles == "" && modelInfo.ConnectionIDs != "" {
			fmt.Printf("DEBUG: Fetching from connections: %s\n", modelInfo.ConnectionIDs)
			connIDs := strings.Split(modelInfo.ConnectionIDs, ",")
			for _, cid := range connIDs {
				cid = strings.TrimSpace(cid)
				if cid == "" { continue }
				var conn Connection
				if DB.Where("id = ?", cid).First(&conn).Error != nil { continue }
				csvPaths, cerr := exportConnectionToCSV(conn, cid)
				if cerr != nil || len(csvPaths) == 0 { continue }
				for _, csvPath := range csvPaths {
					fileID := fmt.Sprintf("conn_%s_%s", cid, strings.TrimSuffix(filepath.Base(csvPath), ".csv"))
					var count int64
					DB.Model(&UploadedFile{}).Where("id = ?", fileID).Count(&count)
					if count == 0 {
						info, serr := os.Stat(csvPath)
						var fsize int64
						if serr == nil { fsize = info.Size() }
						DB.Create(&UploadedFile{
							ID: fileID,
							UserID: modelInfo.UserID,
							Filename: filepath.Base(csvPath),
							Path: csvPath,
							Size: fsize,
							Source: "connection",
						})
					}
					if actualFileID == "" { actualFileID = fileID }
				}
			}
			if actualFileID != "" {
				DB.Table("fine_tuned_models").Where("id = ?", req.FineTunedModel).Update("source_files", actualFileID)
			}
		}
		result, err := callFineTunedModel(req.FineTunedModel, actualFileID, req.Message, modelInfo.ModelPath, userID)
		if err != nil {
			fmt.Printf("Fine-tuned model error: %v\\n", err)
		} else {
			fineTunedResult = result
			fmt.Printf("DEBUG fineTunedResult length: %d\n", len(fineTunedResult))
if strings.Contains(fineTunedResult, "vsRaptors") {
fmt.Println("DEBUG: fineTunedResult CONTAINS vsRaptors!")
} else {
fmt.Println("DEBUG: fineTunedResult does NOT contain vsRaptors")
}
		}
	}

	// Fine-tuned model analysis is already included via callFineTunedModel
	// Get all source file names for the model
	promptFilename := req.Filename
	fmt.Printf("DEBUG REQ: FineTunedModel=%q Filename=%q\n", req.FineTunedModel, req.Filename)
	if req.FineTunedModel != "" {
		var mfn FineTunedModel
		if err := DB.Where("id = ?", req.FineTunedModel).First(&mfn).Error; err == nil {
			fmt.Printf("DEBUG MODEL: SourceName=%q\n", mfn.SourceName)
			if mfn.SourceName != "" {
				promptFilename = mfn.SourceName
			}
		} else {
			fmt.Printf("DEBUG MODEL ERROR: %v\n", err)
		}
	}
	fmt.Printf("DEBUG PROMPT FILENAME: %q\n", promptFilename)
	basePrompt := getSystemPrompt(promptFilename, req.DataContext, "")
	
	// Add Vertical AI Runtime context to LLM prompt
	verticalCtx := GetVerticalContext(userID, req.FineTunedModel)
	if verticalCtx != "" {
		basePrompt += verticalCtx
		fmt.Printf("DEBUG: Vertical context added (%d chars)\n", len(verticalCtx))
	}
	var systemPrompt string
	if fineTunedResult != "" {
		chunks := chunkAnalysis(fineTunedResult, 80000)
		fmt.Printf("DEBUG chunks[0] length: %d\n", len(chunks[0]))
		systemPrompt = basePrompt + "\n\n### Analysis (Part 1/" + fmt.Sprintf("%d", len(chunks)) + "):\n" + chunks[0]
	} else {
		systemPrompt = basePrompt
	}

	// Model-specific history key for compare mode
	historyKey := sessionID
	if req.CompareGroup != "" && req.FineTunedModel != "" {
		historyKey = sessionID + "_" + req.FineTunedModel
	} else if req.CompareGroup != "" {
		historyKey = sessionID + "_" + req.Model
	}
	historyMutex.Lock()
	sess, exists := conversationSessions[historyKey]
	if !exists {
		sess = &ChatSession{Messages: []ChatMessage{}, CreatedAt: time.Now(), LastActive: time.Now(), ExpiresAt: time.Now().Add(sessionTTLMinutes * time.Minute), UserID: userID, SessionID: sessionID, Status: "active"}
		conversationSessions[historyKey] = sess
	}
	if sess.MessageCount >= maxMessagesPerSession {
		historyMutex.Unlock()
		http.Error(w, `{"error":"Session message limit reached (50). Please start a new chat."}`, http.StatusTooManyRequests)
		return
	}
	sess.LastActive = time.Now()
	sess.Messages = append(sess.Messages, ChatMessage{Role: "user", Content: req.Message})
	sess.MessageCount++
	if len(sess.Messages) > maxHistoryTurns*2 {
		sess.Messages = sess.Messages[len(sess.Messages)-maxHistoryTurns*2:]
	}
	history := make([]ChatMessage, len(sess.Messages))
	copy(history, sess.Messages)
	historyMutex.Unlock()

	// ─── Language Layer: Function Calling Mode ───
	llActive, verticalID := IsLanguageLayerActive(userID, req.FineTunedModel)
	if llActive {
		sess.VerticalID = verticalID
		fmt.Printf("[LANGUAGE_LAYER] Active for user=%s vertical=%s model=%s\n", userID, verticalID, req.Model)

		// Resolve provider: user's selected model overrides config
		provider := &LLMProvider{Model: req.Model}
		if isClaudeModel(req.Model) {
			provider.Type = "anthropic"
		} else if strings.HasPrefix(req.Model, "gemini") {
			provider.Type = "gemini"
		} else if strings.HasPrefix(req.Model, "mistral") || strings.HasPrefix(req.Model, "ministral") {
			provider.Type = "mistral"
		} else {
			provider.Type = "openai"
		}

		response, tokens, funcCalls, err := CallLLMWithFunctions(history, systemPrompt, userID, verticalID, req.FineTunedModel, sessionID, provider, w)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		funcCallsJSON := ""
		if len(funcCalls) > 0 {
			fcBytes, _ := json.Marshal(funcCalls)
			funcCallsJSON = string(fcBytes)
		}
		historyMutex.Lock()
		conversationSessions[historyKey].Messages = append(conversationSessions[historyKey].Messages, ChatMessage{Role: "assistant", Content: response})
		historyMutex.Unlock()
		fmt.Printf("[SAVE] sessionID=%s user=%s msg=%s\n", sessionID, userID, req.Message[:min(30, len(req.Message))])
saveMessagesToDB(userID, sessionID, req.Message, response, req.Model, tokens, req.CompareGroup, fmt.Sprintf("%.1fs", time.Since(startTime).Seconds()), req.FineTunedModel)
		if funcCallsJSON != "" {
result := DB.Model(&Message{}).Where("query_id = ? AND role = ?", sessionID, "assistant").Order("created_at desc").Limit(1).Update("function_calls", funcCallsJSON); fmt.Printf("[FUNC_SAVE] rows=%d err=%v\n", result.RowsAffected, result.Error)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"response": response, "model": req.Model, "tokens": tokens, "status": "success", "function_calls": funcCalls})
		return
	}

	// Check if Claude model - use non-streaming
	if isClaudeModel(req.Model) {
		req.Stream = false // Enable streaming for Claude
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
			fmt.Printf("DEBUG LLM Response: %s\n", response)
			if len(response) > 10 {
				conversationSessions[historyKey].Messages = append(conversationSessions[historyKey].Messages, ChatMessage{Role: "assistant", Content: response})
			}
			historyMutex.Unlock()

			// Save to DB
			saveMessagesToDB(userID, sessionID, req.Message, response, req.Model, tokens, req.CompareGroup, fmt.Sprintf("%.1fs", time.Since(startTime).Seconds()), req.FineTunedModel)

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
		fmt.Printf("DEBUG LLM Response: %s\n", response)
		if len(response) > 10 {
			conversationSessions[historyKey].Messages = append(conversationSessions[historyKey].Messages, ChatMessage{Role: "assistant", Content: response})
		}
		historyMutex.Unlock()

		// Save to DB
		saveMessagesToDB(userID, sessionID, req.Message, response, req.Model, tokens, req.CompareGroup, fmt.Sprintf("%.1fs", time.Since(startTime).Seconds()), req.FineTunedModel)

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
		if resp.StatusCode != 200 {
			errBody, _ := io.ReadAll(resp.Body)
			fmt.Printf("DEBUG OpenAI Error: %s\n", string(errBody))
			return
		}
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
		if fullResponse.Len() > 10 {
			conversationSessions[historyKey].Messages = append(conversationSessions[historyKey].Messages, ChatMessage{Role: "assistant", Content: fullResponse.String()})
		}
		historyMutex.Unlock()

		// Save to DB
		saveMessagesToDB(userID, sessionID, req.Message, fullResponse.String(), req.Model, len(fullResponse.String())/4, req.CompareGroup, fmt.Sprintf("%.1fs", time.Since(startTime).Seconds()), req.FineTunedModel)

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
	if resp.StatusCode != 200 {
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Printf("DEBUG OpenAI Error: %s\n", string(errBody))
		return
	}
	if err != nil {
		http.Error(w, "Failed to call OpenAI", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("DEBUG OpenAI Response (first 500 chars): %s\n", string(body)[:min(len(body), 500)])
	var openAIResp OpenAIResponse
	json.Unmarshal(body, &openAIResp)

	if len(openAIResp.Choices) == 0 {
		http.Error(w, "No response", http.StatusInternalServerError)
		return
	}

	assistantMsg := openAIResp.Choices[0].Message.Content

	historyMutex.Lock()
	if len(assistantMsg) > 10 {
		conversationSessions[historyKey].Messages = append(conversationSessions[historyKey].Messages, ChatMessage{Role: "assistant", Content: assistantMsg})
	}
	historyMutex.Unlock()

	// Save to DB
	saveMessagesToDB(userID, sessionID, req.Message, assistantMsg, req.Model, openAIResp.Usage.TotalTokens, req.CompareGroup, fmt.Sprintf("%.1fs", time.Since(startTime).Seconds()), req.FineTunedModel)

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
	delete(conversationSessions, req.SessionID)
	historyMutex.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
}

// callFineTunedModel calls the Flask server for fine-tuned model analysis
func callFineTunedModel(modelID string, fileID string, message string, modelPath string, userID string) (string, error) {
	flaskURL := GetFlaskURL()

	payload := map[string]interface{}{
		"model_id":   modelID,
		"file_id":    fileID,
		"message":    message,
		"model_path": modelPath,
		"user_id":    userID,
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
