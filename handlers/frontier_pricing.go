package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type FrontierPrice struct {
	ID           string    `gorm:"primaryKey" json:"id"`
	Provider     string    `gorm:"index" json:"provider"`
	Model        string    `gorm:"uniqueIndex" json:"model"`
	InputPer1M   float64   `json:"input_per_1m"`
	OutputPer1M  float64   `json:"output_per_1m"`
	Source       string    `json:"source"`
	UpdatedAt    time.Time `json:"updated_at"`
}

func GetFrontierRate(model string) (inputPer1M, outputPer1M float64) {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return getEnvPrice("FRONTIER_DEFAULT_INPUT", 3.0), getEnvPrice("FRONTIER_DEFAULT_OUTPUT", 15.0)
	}

	if DB != nil {
		var fp FrontierPrice
		if err := DB.Where("LOWER(model) = ?", m).First(&fp).Error; err == nil {
			return fp.InputPer1M, fp.OutputPer1M
		}
		if err := DB.Where("? LIKE '%' || LOWER(model) || '%'", m).Order("LENGTH(model) DESC").First(&fp).Error; err == nil {
			return fp.InputPer1M, fp.OutputPer1M
		}
	}

	return getEnvPrice("FRONTIER_DEFAULT_INPUT", 3.0), getEnvPrice("FRONTIER_DEFAULT_OUTPUT", 15.0)
}

func FetchFrontierPricesFromLiteLLM() error {
	url := os.Getenv("LITELLM_PRICES_URL")
	if url == "" {
		url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("fetch litellm: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read litellm: %w", err)
	}

	var raw map[string]map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return fmt.Errorf("parse litellm: %w", err)
	}

	count := 0
	now := time.Now()
	for model, data := range raw {
		if model == "sample_spec" {
			continue
		}
		provider, _ := data["litellm_provider"].(string)
		inputCost, _ := data["input_cost_per_token"].(float64)
		outputCost, _ := data["output_cost_per_token"].(float64)
		if inputCost == 0 && outputCost == 0 {
			continue
		}

		fp := FrontierPrice{
			ID:          fmt.Sprintf("fp-%s", strings.ReplaceAll(model, "/", "-")),
			Provider:    provider,
			Model:       model,
			InputPer1M:  inputCost * 1_000_000,
			OutputPer1M: outputCost * 1_000_000,
			Source:      "litellm",
			UpdatedAt:   now,
		}
		if DB != nil {
			DB.Save(&fp)
			count++
		}
	}

	log.Printf("[FRONTIER_PRICING] synced %d models from litellm", count)
	return nil
}

func StartFrontierPricingSync() {
	go func() {
		if err := FetchFrontierPricesFromLiteLLM(); err != nil {
			log.Printf("[FRONTIER_PRICING] initial sync failed: %v", err)
		}
		ticker := time.NewTicker(24 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			if err := FetchFrontierPricesFromLiteLLM(); err != nil {
				log.Printf("[FRONTIER_PRICING] periodic sync failed: %v", err)
			}
		}
	}()
}
