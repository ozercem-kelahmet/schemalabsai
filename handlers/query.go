package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

type CreateQueryRequest struct {
	Name            string   `json:"name"`
	Model           string   `json:"model"`
	DataSources     []string `json:"dataSources"`
	FileID          string   `json:"fileId"`
	IsTraining      bool     `json:"isTraining"`
	HasModel        bool     `json:"hasModel"`
TrainingFailed  bool     `json:"trainingFailed"`
	TrainingModelID *string  `json:"trainingModelId"`
	ModelName       string   `json:"modelName"`
	ModelAccuracy   float64  `json:"modelAccuracy"`
	SourceCsvName   string   `json:"sourceCsvName"`
	SourceFiles     string   `json:"sourceFiles"`
}

type QueryResponse struct {
	ID              string   `json:"id"`
	Name            string   `json:"name"`
	Model           string   `json:"model"`
	DataSources     []string `json:"dataSources"`
	FileID          string   `json:"fileId"`
	IsTraining      bool     `json:"isTraining"`
	HasModel        bool     `json:"hasModel"`
TrainingFailed  bool     `json:"trainingFailed"`
	TrainingModelID *string  `json:"trainingModelId"`
	ModelName       string   `json:"modelName"`
	ModelAccuracy   float64  `json:"modelAccuracy"`
	SourceCsvName   string   `json:"sourceCsvName"`
	SourceFiles     string   `json:"sourceFiles"`
	CreatedAt       string   `json:"createdAt"`
}

func CreateQueryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fmt.Printf("USER ID FROM HEADER: %q\n", userID)
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req CreateQueryRequest
	json.NewDecoder(r.Body).Decode(&req)

	// DEBUG LOG
	fmt.Printf("CREATE QUERY REQUEST: name=%q, model=%q, isTraining=%v, hasModel=%v, dataSources=%v\n",
		req.Name, req.Model, req.IsTraining, req.HasModel, req.DataSources)

	queryID := uuid.New().String()
	query := Query{
		ID:              queryID,
		Name:            req.Name,
		Model:           req.Model,
		UserID:          userID,
		IsTraining:      req.IsTraining,
		HasModel:        req.HasModel,
		TrainingModelID: req.TrainingModelID,
		FileID:          req.FileID,
		ModelName:       req.ModelName,
		ModelAccuracy:   req.ModelAccuracy,
		SourceCsvName:   req.SourceCsvName,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	if err := DB.Create(&query).Error; err != nil {
		fmt.Printf("DB CREATE ERROR: %v\n", err)
		http.Error(w, "Failed to create query", http.StatusInternalServerError)
		return
	}

	fmt.Printf("QUERY CREATED: id=%s, name=%s\n", queryID, req.Name)

	// Link files to query - use FileID if provided
	if req.FileID != "" {
		if err := DB.Create(&QueryFile{QueryID: queryID, FileID: req.FileID}).Error; err != nil { fmt.Printf("DB ERROR: %v\n", err) }
		fmt.Printf("LINKED FILE: query=%s, file=%s\n", queryID, req.FileID)
	} else {
		for _, fileID := range req.DataSources {
			DB.Create(&QueryFile{QueryID: queryID, FileID: fileID})
		}
	}
	// Get source_files from fine_tuned_model
	sourceFiles := ""
	if req.TrainingModelID != nil && *req.TrainingModelID != "" {
		var model FineTunedModel
		if err := DB.Where("id = ?", *req.TrainingModelID).First(&model).Error; err == nil {
			sourceFiles = model.SourceFiles
			// If source_files empty but connection_ids exists, fetch from connection
			if sourceFiles == "" && model.ConnectionIDs != "" {
				connIDs := strings.Split(model.ConnectionIDs, ",")
				var allFileIDs []string
				for _, cid := range connIDs {
					cid = strings.TrimSpace(cid)
					if cid == "" { continue }
					var conn Connection
					if DB.Where("id = ?", cid).First(&conn).Error != nil { continue }
					csvPaths, err := exportConnectionToCSV(conn, cid)
					if err != nil || len(csvPaths) == 0 { continue }
					for _, csvPath := range csvPaths {
						fileID := fmt.Sprintf("conn_%s_%s", cid, strings.TrimSuffix(filepath.Base(csvPath), ".csv"))
						var count int64
						DB.Model(&UploadedFile{}).Where("id = ?", fileID).Count(&count)
						if count == 0 {
							info, err := os.Stat(csvPath)
							var fsize int64
							if err == nil { fsize = info.Size() }
							DB.Create(&UploadedFile{
								ID: fileID,
								UserID: model.UserID,
								Filename: filepath.Base(csvPath),
								Path: csvPath,
								Size: fsize,
								Source: "connection",
							})
						}
						allFileIDs = append(allFileIDs, fileID)
					}
				}
				if len(allFileIDs) > 0 {
					sourceFiles = strings.Join(allFileIDs, ",")
					DB.Model(&model).Update("source_files", sourceFiles)
				}
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(QueryResponse{
		ID:              queryID,
		Name:            req.Name,
		Model:           req.Model,
		DataSources:     req.DataSources,
		IsTraining:      req.IsTraining,
		HasModel:        req.HasModel,
		TrainingModelID: req.TrainingModelID,
		FileID:          req.FileID,
		ModelName:       req.ModelName,
		ModelAccuracy:   req.ModelAccuracy,
		SourceCsvName:   req.SourceCsvName,
		SourceFiles:     sourceFiles,
		CreatedAt:       query.CreatedAt.Format(time.RFC3339),
	})
}

func ListQueriesHandler(w http.ResponseWriter, r *http.Request) {
startTime := time.Now()
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fmt.Printf("USER ID FROM HEADER: %q\n", userID)
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	modelID := r.URL.Query().Get("model_id")
var queries []Query
	if modelID != "" {
	DB.Where("user_id = ? AND training_model_id = ?", userID, modelID).Order("updated_at desc").Find(&queries)
} else {
	DB.Where("user_id = ?", userID).Order("updated_at desc").Limit(50).Find(&queries)
}

var response []QueryResponse
for _, q := range queries {
response = append(response, QueryResponse{
ID:              q.ID,
Name:            q.Name,
Model:           q.Model,
DataSources:     []string{},
IsTraining:      q.IsTraining,
HasModel:        q.HasModel,
TrainingFailed:  q.TrainingFailed,
TrainingModelID: q.TrainingModelID,
ModelName:       q.ModelName,
ModelAccuracy:   q.ModelAccuracy,
SourceCsvName:   q.SourceCsvName,
CreatedAt:       q.CreatedAt.Format(time.RFC3339),
FileID:          q.FileID,
SourceFiles:     "",
})
}
	w.Header().Set("Content-Type", "application/json")
	fmt.Printf("ListQueriesHandler took %v for %d queries\n", time.Since(startTime), len(response))
json.NewEncoder(w).Encode(map[string]interface{}{"queries": response})
}

func DeleteQueryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fmt.Printf("USER ID FROM HEADER: %q\n", userID)
	queryID := r.URL.Query().Get("id")

	if userID == "" || queryID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	DB.Where("query_id = ?", queryID).Delete(&Message{})
	DB.Where("query_id = ?", queryID).Delete(&QueryFile{})
	DB.Where("id = ? AND user_id = ?", queryID, userID).Delete(&Query{})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

func GetMessagesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fmt.Printf("USER ID FROM HEADER: %q\n", userID)
	queryID := r.URL.Query().Get("query_id")
	modelID := r.URL.Query().Get("model_id")

	if userID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	var messages []Message

	if modelID != "" {
		fmt.Printf("DEBUG: Getting messages for model_id: %s\n", modelID)
		var queryIDs []string
		var model FineTunedModel; modelName := modelID; if err := DB.Where("id = ?", modelID).First(&model).Error; err == nil && model.ModelPath != "" { modelName = strings.TrimSuffix(strings.TrimPrefix(model.ModelPath, "../checkpoints/"), ".pt") }; DB.Model(&Query{}).Where("user_id = ? AND (training_model_id = ? OR training_model_id = ?)", userID, modelID, modelName).Pluck("id", &queryIDs)
		fmt.Printf("DEBUG: Found %d queries for model\n", len(queryIDs))
		if len(queryIDs) > 0 {
			DB.Where("query_id IN ? AND user_id = ?", queryIDs, userID).Order("created_at asc").Find(&messages)
		}
	} else if queryID != "" {
		DB.Where("query_id = ? AND user_id = ?", queryID, userID).Order("created_at asc").Find(&messages)
	} else {
		http.Error(w, "Bad request: query_id or model_id required", http.StatusBadRequest)
		return
	}

	fmt.Printf("DEBUG: Returning %d messages\n", len(messages))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"messages": messages})
}

func UpdateQueryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fmt.Printf("USER ID FROM HEADER: %q\n", userID)
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		ID              string  `json:"id"`
		Name            string  `json:"name,omitempty"`
		IsTraining      *bool   `json:"isTraining,omitempty"`
		HasModel        *bool   `json:"hasModel,omitempty"`
		TrainingModelID *string `json:"trainingModelId,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var query Query
	if err := DB.Where("id = ? AND user_id = ?", req.ID, userID).First(&query).Error; err != nil {
		http.Error(w, "Query not found", http.StatusNotFound)
		return
	}

	if req.Name != "" {
		query.Name = req.Name
	}
	if req.IsTraining != nil {
		query.IsTraining = *req.IsTraining
	}
	if req.HasModel != nil {
		query.HasModel = *req.HasModel
	}
	if req.TrainingModelID != nil {
		query.TrainingModelID = req.TrainingModelID
	}
	query.UpdatedAt = time.Now()
	DB.Save(&query)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Query updated"})
}
