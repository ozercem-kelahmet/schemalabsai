package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
)

type PredictRequest struct {
	Values [][]float64 `json:"values"`
}

type PredictResponse struct {
	Predictions   []int       `json:"predictions"`
	Confidences   []float64   `json:"confidences"`
	Probabilities [][]float64 `json:"probabilities"`
	Status        string      `json:"status"`
}

func PredictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	jsonData, _ := json.Marshal(req)

	resp, err := http.Post(
		GetFlaskURL()+"/predict",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		http.Error(w, "Flask server error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(GetFlaskURL()+"/health")
	if err != nil {
		http.Error(w, "Flask server down", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}

func ModelInfoHandler(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(GetFlaskURL()+"/model/info")
	if err != nil {
		http.Error(w, "Flask server down", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}
