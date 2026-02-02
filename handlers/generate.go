package handlers

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

type GenerateRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Rows        int    `json:"rows"`
	Columns     int    `json:"columns"`
	Vertical    string `json:"vertical"`
	Prompt      string `json:"prompt"`
	UsePython   bool   `json:"use_python"`
	PythonCode  string `json:"python_code"`
}

func GenerateDatasetHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Generate unique file ID
	fileID := uuid.New().String()
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s.csv", sanitizeFilename(req.Name), timestamp)
	destPath := filepath.Join("./uploads", filename)

	var err error
	if req.UsePython {
		err = generateWithPython(req, destPath)
	} else {
		err = generateSyntheticData(req, destPath)
	}

	if err != nil {
		http.Error(w, "Generation failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Get file info
	fileInfo, _ := os.Stat(destPath)
	size := int64(0)
	if fileInfo != nil {
		size = fileInfo.Size()
	}

	// Count rows and columns
	rowCount, columns := countCSVStats(destPath)

	// Save to database
	if DB != nil {
		uploadedFile := UploadedFile{
			ID:        fileID,
			Filename:  filename,
			Path:      destPath,
			Size:      size,
			UserID:    userID,
			CreatedAt: time.Now(),
			Columns:   columns,
			RowCount:  rowCount,
		Vertical:  req.Vertical,
		Source:    "generated",
		}
		DB.Create(&uploadedFile)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "success",
		"file_id":  fileID,
		"filename": filename,
		"rows":     rowCount,
		"columns":  columns,
	})
}

func generateWithPython(req GenerateRequest, destPath string) error {
	// Create temp Python script
	tempScript := filepath.Join(os.TempDir(), "generate_"+uuid.New().String()+".py")

	// Wrap user code to ensure it outputs to the correct path
	wrappedCode := fmt.Sprintf(`
import pandas as pd
import numpy as np
from faker import Faker
import random

# User's code
%s

# Save to CSV if df exists
if 'df' in dir():
    df.to_csv('%s', index=False)
`, req.PythonCode, destPath)

	if err := os.WriteFile(tempScript, []byte(wrappedCode), 0644); err != nil {
		return err
	}
	defer os.Remove(tempScript)

	// Execute Python script
	cmd := exec.Command("python3", tempScript)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("python error: %s - %s", err.Error(), string(output))
	}

	return nil
}

func generateSyntheticData(req GenerateRequest, destPath string) error {
	rand.Seed(time.Now().UnixNano())

	// Get column definitions based on vertical
	columnDefs := getColumnDefinitions(req.Vertical, req.Columns, req.Prompt)

	file, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	headers := make([]string, len(columnDefs))
	for i, col := range columnDefs {
		headers[i] = col.Name
	}
	writer.Write(headers)

	// Generate rows
	for i := 0; i < req.Rows; i++ {
		row := make([]string, len(columnDefs))
		for j, col := range columnDefs {
			row[j] = generateValue(col, i)
		}
		writer.Write(row)
	}

	return nil
}

type ColumnDef struct {
	Name    string
	Type    string // string, int, float, date, category, bool
	Min     float64
	Max     float64
	Options []string
}

func getColumnDefinitions(vertical string, numCols int, _ string) []ColumnDef {
	// Vertical-specific column templates
	templates := map[string][]ColumnDef{
		"finance": {
			{Name: "transaction_id", Type: "id"},
			{Name: "customer_id", Type: "id"},
			{Name: "amount", Type: "float", Min: 10, Max: 10000},
			{Name: "transaction_date", Type: "date"},
			{Name: "category", Type: "category", Options: []string{"retail", "food", "travel", "utilities", "entertainment"}},
			{Name: "merchant", Type: "company"},
			{Name: "payment_method", Type: "category", Options: []string{"credit_card", "debit_card", "bank_transfer", "cash"}},
			{Name: "is_fraud", Type: "bool_weighted", Min: 0.02}, // 2% fraud rate
			{Name: "risk_score", Type: "float", Min: 0, Max: 1},
			{Name: "balance", Type: "float", Min: 0, Max: 50000},
		},
		"healthcare": {
			{Name: "patient_id", Type: "id"},
			{Name: "age", Type: "int", Min: 18, Max: 90},
			{Name: "gender", Type: "category", Options: []string{"M", "F"}},
			{Name: "diagnosis_code", Type: "category", Options: []string{"J06", "I10", "E11", "M54", "F32", "K21"}},
			{Name: "admission_date", Type: "date"},
			{Name: "discharge_date", Type: "date"},
			{Name: "treatment_cost", Type: "float", Min: 100, Max: 50000},
			{Name: "insurance_type", Type: "category", Options: []string{"private", "public", "none"}},
			{Name: "readmission", Type: "bool_weighted", Min: 0.15},
			{Name: "satisfaction_score", Type: "int", Min: 1, Max: 5},
		},
		"e-commerce": {
			{Name: "order_id", Type: "id"},
			{Name: "customer_id", Type: "id"},
			{Name: "product_id", Type: "id"},
			{Name: "product_category", Type: "category", Options: []string{"electronics", "clothing", "home", "beauty", "sports"}},
			{Name: "quantity", Type: "int", Min: 1, Max: 10},
			{Name: "unit_price", Type: "float", Min: 5, Max: 500},
			{Name: "total_amount", Type: "calculated"},
			{Name: "order_date", Type: "date"},
			{Name: "shipping_method", Type: "category", Options: []string{"standard", "express", "same_day"}},
			{Name: "returned", Type: "bool_weighted", Min: 0.08},
		},
		"marketing": {
			{Name: "campaign_id", Type: "id"},
			{Name: "customer_id", Type: "id"},
			{Name: "channel", Type: "category", Options: []string{"email", "social", "search", "display", "affiliate"}},
			{Name: "impressions", Type: "int", Min: 100, Max: 100000},
			{Name: "clicks", Type: "int", Min: 0, Max: 5000},
			{Name: "conversions", Type: "int", Min: 0, Max: 500},
			{Name: "spend", Type: "float", Min: 10, Max: 10000},
			{Name: "revenue", Type: "float", Min: 0, Max: 50000},
			{Name: "campaign_date", Type: "date"},
			{Name: "target_segment", Type: "category", Options: []string{"new", "returning", "vip", "at_risk"}},
		},
		"hr": {
			{Name: "employee_id", Type: "id"},
			{Name: "department", Type: "category", Options: []string{"engineering", "sales", "marketing", "hr", "finance", "operations"}},
			{Name: "job_level", Type: "category", Options: []string{"junior", "mid", "senior", "lead", "manager"}},
			{Name: "hire_date", Type: "date"},
			{Name: "salary", Type: "float", Min: 30000, Max: 200000},
			{Name: "performance_score", Type: "float", Min: 1, Max: 5},
			{Name: "tenure_months", Type: "int", Min: 1, Max: 240},
			{Name: "is_remote", Type: "bool_weighted", Min: 0.3},
			{Name: "training_hours", Type: "int", Min: 0, Max: 100},
			{Name: "churned", Type: "bool_weighted", Min: 0.12},
		},
		"operations": {
			{Name: "order_id", Type: "id"},
			{Name: "product_id", Type: "id"},
			{Name: "warehouse", Type: "category", Options: []string{"WH-A", "WH-B", "WH-C", "WH-D"}},
			{Name: "quantity", Type: "int", Min: 1, Max: 1000},
			{Name: "processing_time_hours", Type: "float", Min: 0.5, Max: 72},
			{Name: "shipping_cost", Type: "float", Min: 5, Max: 500},
			{Name: "delivery_date", Type: "date"},
			{Name: "on_time", Type: "bool_weighted", Min: 0.92},
			{Name: "defect_rate", Type: "float", Min: 0, Max: 0.05},
			{Name: "supplier", Type: "category", Options: []string{"SUP-001", "SUP-002", "SUP-003", "SUP-004", "SUP-005"}},
		},
	}

	// Get template for vertical or use default
	cols, ok := templates[vertical]
	if !ok {
		cols = templates["finance"]
	}

	// Adjust to requested number of columns
	if numCols > 0 && numCols < len(cols) {
		cols = cols[:numCols]
	}

	return cols
}

func generateValue(col ColumnDef, rowIndex int) string {
	switch col.Type {
	case "id":
		return fmt.Sprintf("%s_%d", strings.ToUpper(col.Name[:3]), 10000+rowIndex)
	case "int":
		return strconv.Itoa(int(col.Min) + rand.Intn(int(col.Max-col.Min+1)))
	case "float":
		val := col.Min + rand.Float64()*(col.Max-col.Min)
		return fmt.Sprintf("%.2f", val)
	case "date":
		days := rand.Intn(365 * 2)
		date := time.Now().AddDate(0, 0, -days)
		return date.Format("2006-01-02")
	case "category":
		return col.Options[rand.Intn(len(col.Options))]
	case "bool":
		if rand.Float64() > 0.5 {
			return "true"
		}
		return "false"
	case "bool_weighted":
		if rand.Float64() < col.Min {
			return "true"
		}
		return "false"
	case "company":
		companies := []string{"Amazon", "Walmart", "Target", "Costco", "BestBuy", "HomeDepot", "Starbucks", "McDonalds"}
		return companies[rand.Intn(len(companies))]
	case "calculated":
		// This would need context from other columns
		return fmt.Sprintf("%.2f", rand.Float64()*1000)
	default:
		return fmt.Sprintf("value_%d", rowIndex)
	}
}

func countCSVStats(filepath string) (int, string) {
	file, err := os.Open(filepath)
	if err != nil {
		return 0, ""
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil || len(records) == 0 {
		return 0, ""
	}

	headers := strings.Join(records[0], ",")
	rowCount := len(records) - 1 // Exclude header

	return rowCount, headers
}
