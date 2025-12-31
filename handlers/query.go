package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type CreateQueryRequest struct {
	Name            string   `json:"name"`
	Model           string   `json:"model"`
	DataSources     []string `json:"data_sources"`
	IsTraining      bool     `json:"is_training"`
	HasModel        bool     `json:"has_model"`
	TrainingModelID *string  `json:"training_model_id"`
}

type QueryResponse struct {
	ID              string   `json:"id"`
	Name            string   `json:"name"`
	Model           string   `json:"model"`
	DataSources     []string `json:"data_sources"`
	IsTraining      bool     `json:"is_training"`
	HasModel        bool     `json:"has_model"`
	TrainingModelID *string  `json:"training_model_id"`
	CreatedAt       string   `json:"created_at"`
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
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	if err := DB.Create(&query).Error; err != nil {
		fmt.Printf("DB CREATE ERROR: %v\n", err)
		http.Error(w, "Failed to create query", http.StatusInternalServerError)
		return
	}

	fmt.Printf("QUERY CREATED: id=%s, name=%s\n", queryID, req.Name)

	// Link files to query
	for _, fileID := range req.DataSources {
		DB.Create(&QueryFile{QueryID: queryID, FileID: fileID})
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
		CreatedAt:       query.CreatedAt.Format(time.RFC3339),
	})
}

func ListQueriesHandler(w http.ResponseWriter, r *http.Request) {
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

	var queries []Query
	DB.Where("user_id = ?", userID).Order("updated_at desc").Find(&queries)

	var response []QueryResponse
	for _, q := range queries {
		var queryFiles []QueryFile
		DB.Where("query_id = ?", q.ID).Find(&queryFiles)

		fileIDs := make([]string, len(queryFiles))
		for i, qf := range queryFiles {
			fileIDs[i] = qf.FileID
		}

		response = append(response, QueryResponse{
			ID:              q.ID,
			Name:            q.Name,
			Model:           q.Model,
			DataSources:     fileIDs,
			IsTraining:      q.IsTraining,
			HasModel:        q.HasModel,
			TrainingModelID: q.TrainingModelID,
			CreatedAt:       q.CreatedAt.Format(time.RFC3339),
		})
	}

	w.Header().Set("Content-Type", "application/json")
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

	if userID == "" || queryID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	var messages []Message
	DB.Where("query_id = ? AND user_id = ?", queryID, userID).Order("created_at asc").Find(&messages)

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
