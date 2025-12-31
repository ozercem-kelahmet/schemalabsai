package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)


func ListFoldersHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var folders []Folder
	DB.Where("user_id = ?", userID).Order("created_at asc").Find(&folders)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"folders": folders})
}

func CreateFolderHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var req struct {
		Name string `json:"name"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	folder := Folder{
		ID:        uuid.New().String(),
		Name:      req.Name,
		UserID:    userID,
		CreatedAt: time.Now(),
	}

	DB.Create(&folder)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(folder)
}

func UpdateFolderHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	folderID := r.URL.Query().Get("id")

	var req struct {
		Name string `json:"name"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	DB.Model(&Folder{}).Where("id = ? AND user_id = ?", folderID, userID).Update("name", req.Name)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
}

func DeleteFolderHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	folderID := r.URL.Query().Get("id")

	DB.Model(&UploadedFile{}).Where("folder_id = ? AND user_id = ?", folderID, userID).Update("folder_id", nil)
	DB.Where("id = ? AND user_id = ?", folderID, userID).Delete(&Folder{})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

func MoveFileToFolderHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	var req struct {
		FileID   string  `json:"file_id"`
		FolderID *string `json:"folder_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	DB.Model(&UploadedFile{}).Where("id = ? AND user_id = ?", req.FileID, userID).Update("folder_id", req.FolderID)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "moved"})
}
