package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type SearchRequest struct {
	Query string `json:"query"`
	Index string `json:"index"`
	Size  int    `json:"size"`
	From  int    `json:"from"`
}

type SearchResult struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Type     string                 `json:"type"`
	Size     int64                  `json:"size"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Total   int            `json:"total"`
	Took    int            `json:"took"`
}

func getElasticURL() string {
	url := os.Getenv("ELASTICSEARCH_URL")
	if url == "" {
		url = "http://localhost:9200"
	}
	return url
}

// SearchHandler - Elasticsearch ile arama
func SearchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		json.NewEncoder(w).Encode(SearchResponse{Results: []SearchResult{}, Total: 0})
		return
	}

	if req.Size == 0 {
		req.Size = 20
	}
	if req.Index == "" {
		req.Index = "data_sources"
	}

	userID := r.Header.Get("X-User-ID")

	// Elasticsearch query
	esQuery := map[string]interface{}{
		"query": map[string]interface{}{
			"bool": map[string]interface{}{
				"must": []map[string]interface{}{
					{
						"multi_match": map[string]interface{}{
							"query":     req.Query,
							"fields":    []string{"name^3", "type", "filename", "description"},
							"fuzziness": "AUTO",
						},
					},
				},
				"filter": []map[string]interface{}{
					{
						"term": map[string]interface{}{
							"user_id": userID,
						},
					},
				},
			},
		},
		"size": req.Size,
		"from": req.From,
		"highlight": map[string]interface{}{
			"fields": map[string]interface{}{
				"name": map[string]interface{}{},
			},
		},
	}

	queryBytes, _ := json.Marshal(esQuery)
	esURL := fmt.Sprintf("%s/%s/_search", getElasticURL(), req.Index)

	esReq, err := http.NewRequest("POST", esURL, bytes.NewBuffer(queryBytes))
	if err != nil {
		// Elasticsearch bağlantısı yoksa fallback
		fallbackSearch(w, r, req.Query, userID)
		return
	}
	esReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(esReq)
	if err != nil {
		// Elasticsearch bağlantısı yoksa fallback
		fallbackSearch(w, r, req.Query, userID)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		fallbackSearch(w, r, req.Query, userID)
		return
	}

	body, _ := io.ReadAll(resp.Body)

	var esResp struct {
		Took int `json:"took"`
		Hits struct {
			Total struct {
				Value int `json:"value"`
			} `json:"total"`
			Hits []struct {
				ID     string                 `json:"_id"`
				Score  float64                `json:"_score"`
				Source map[string]interface{} `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.Unmarshal(body, &esResp); err != nil {
		fallbackSearch(w, r, req.Query, userID)
		return
	}

	results := make([]SearchResult, 0)
	for _, hit := range esResp.Hits.Hits {
		name, _ := hit.Source["name"].(string)
		if name == "" {
			name, _ = hit.Source["filename"].(string)
		}
		fileType, _ := hit.Source["type"].(string)
		size, _ := hit.Source["size"].(float64)

		results = append(results, SearchResult{
			ID:       hit.ID,
			Name:     name,
			Type:     fileType,
			Size:     int64(size),
			Score:    hit.Score,
			Metadata: hit.Source,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{
		Results: results,
		Total:   esResp.Hits.Total.Value,
		Took:    esResp.Took,
	})
}

// fallbackSearch - Elasticsearch yoksa PostgreSQL'den arama
func fallbackSearch(w http.ResponseWriter, _ *http.Request, query string, userID string) {
	db := DB
	if db == nil {
		http.Error(w, "Database connection error", http.StatusInternalServerError)
		return
	}

	query = strings.ToLower(query)

	var files []UploadedFile
	db.Where("user_id = ? AND LOWER(filename) LIKE ?", userID, "%"+query+"%").
		Order("created_at DESC").
		Limit(50).
		Find(&files)

	results := make([]SearchResult, 0)
	for _, f := range files {
		results = append(results, SearchResult{
			ID:    f.ID,
			Name:  f.Filename,
			Type:  getFileType(f.Filename),
			Size:  f.Size,
			Score: 1.0,
			Metadata: map[string]interface{}{
				"path":       f.Path,
				"created_at": f.CreatedAt,
				"folder_id":  f.FolderID,
			},
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{
		Results: results,
		Total:   len(results),
		Took:    0,
	})
}

func getFileType(filename string) string {
	lower := strings.ToLower(filename)
	if strings.HasSuffix(lower, ".csv") {
		return "csv"
	} else if strings.HasSuffix(lower, ".json") {
		return "json"
	} else if strings.HasSuffix(lower, ".parquet") {
		return "parquet"
	} else if strings.HasSuffix(lower, ".xlsx") || strings.HasSuffix(lower, ".xls") {
		return "excel"
	} else if strings.HasSuffix(lower, ".pdf") {
		return "pdf"
	}
	return "file"
}

// IndexDataSource - Yeni dosya yüklendiğinde Elasticsearch'e index'le
func IndexDataSource(fileID, filename, userID string, size int64, folderID *string) error {
	esURL := fmt.Sprintf("%s/data_sources/_doc/%s", getElasticURL(), fileID)

	doc := map[string]interface{}{
		"id":        fileID,
		"name":      filename,
		"filename":  filename,
		"type":      getFileType(filename),
		"size":      size,
		"user_id":   userID,
		"folder_id": folderID,
	}

	docBytes, _ := json.Marshal(doc)
	req, err := http.NewRequest("PUT", esURL, bytes.NewBuffer(docBytes))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// DeleteFromIndex - Dosya silindiğinde Elasticsearch'ten kaldır
func DeleteFromIndex(fileID string) error {
	esURL := fmt.Sprintf("%s/data_sources/_doc/%s", getElasticURL(), fileID)

	req, err := http.NewRequest("DELETE", esURL, nil)
	if err != nil {
		return err
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}
