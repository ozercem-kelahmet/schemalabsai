package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

func getEnvInt(key string, defaultVal int64) int64 {
	if val := os.Getenv(key); val != "" {
		var v int64
		fmt.Sscanf(val, "%d", &v)
		if v > 0 {
			return v
		}
	}
	return defaultVal
}

type UploadedFile struct {
	ID        string    `gorm:"primaryKey" json:"file_id"`
	Filename  string    `json:"filename"`
	Path      string    `json:"path"`
	Size      int64     `json:"size"`
	UserID    string    `json:"user_id"`
	FolderID  *string   `json:"folder_id"`
	CreatedAt time.Time `json:"created_at"`
}

type UploadResponse struct {
	FileID   string `json:"file_id"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

func UploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Failed to read file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	maxFileSizeMB := getEnvInt("MAX_FILE_SIZE_MB", 50)
	maxFileSize := maxFileSizeMB * 1024 * 1024
	if header.Size > maxFileSize {
		http.Error(w, fmt.Sprintf("File too large. Max size: %dMB", maxFileSizeMB), http.StatusBadRequest)
		return
	}

	maxTotalMB := getEnvInt("MAX_TOTAL_STORAGE_MB", 1024)
	maxTotalSize := maxTotalMB * 1024 * 1024
	var totalUsed int64
	if userID != "" && DB != nil {
		DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalUsed)
	}
	if totalUsed + header.Size > maxTotalSize {
		http.Error(w, fmt.Sprintf("Storage limit exceeded. Max: %dMB, Used: %dMB", maxTotalMB, totalUsed/(1024*1024)), http.StatusBadRequest)
		return
	}

	ext := strings.ToLower(filepath.Ext(header.Filename))
	allowed := map[string]bool{".csv": true, ".xlsx": true, ".xls": true, ".json": true, ".parquet": true, ".txt": true, ".pdf": true}
	if !allowed[ext] {
		http.Error(w, "File type not supported", http.StatusBadRequest)
		return
	}

	fileID := uuid.New().String()
	uploadDir := "./uploads"
	os.MkdirAll(uploadDir, 0755)

	// Get base name without extension
	baseName := strings.TrimSuffix(header.Filename, ext)
	
	// Add date timestamp
	dateStr := time.Now().Format("20060102_150405")
	
	// Check for existing files with same base name and get version
	version := 1
	if userID != "" && DB != nil {
		var existingFiles []UploadedFile
		// Find files that start with same base name
		DB.Where("user_id = ? AND filename LIKE ?", userID, baseName+"%").Find(&existingFiles)
		
		if len(existingFiles) > 0 {
			// Find highest version
			for _, f := range existingFiles {
				// Check if filename contains _v followed by number
				fname := strings.TrimSuffix(f.Filename, filepath.Ext(f.Filename))
				if strings.Contains(fname, "_v") {
					parts := strings.Split(fname, "_v")
					if len(parts) > 1 {
						var v int
						fmt.Sscanf(parts[len(parts)-1], "%d", &v)
						if v >= version {
							version = v + 1
						}
					}
				} else {
					// File exists without version, so next should be v2
					if version == 1 {
						version = 2
					}
				}
			}
		}
	}

	// Build final filename: baseName_YYYYMMDD_HHMMSS_vN.ext
	var finalFilename string
	if version > 1 {
		finalFilename = fmt.Sprintf("%s_%s_v%d%s", baseName, dateStr, version, ext)
	} else {
		finalFilename = fmt.Sprintf("%s_%s%s", baseName, dateStr, ext)
	}

	destFilename := fileID + "_" + finalFilename
	destPath := filepath.Join(uploadDir, destFilename)

	dest, err := os.Create(destPath)
	if err != nil {
		http.Error(w, "Failed to save file", http.StatusInternalServerError)
		return
	}
	defer dest.Close()

	size, err := io.Copy(dest, file)
	if err != nil {
		http.Error(w, "Failed to write file", http.StatusInternalServerError)
		return
	}

	// Save to database if user is logged in
	if userID != "" && DB != nil {
		uploadedFile := UploadedFile{
			ID:        fileID,
			Filename:  finalFilename,
			Path:      destPath,
			Size:      size,
			UserID:    userID,
			CreatedAt: time.Now(),
		}
		DB.Create(&uploadedFile)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(UploadResponse{
		FileID:   fileID,
		Filename: finalFilename,
		Size:     size,
	})
}

func GetUploadedFilesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")

	if userID != "" && DB != nil {
		var files []UploadedFile
		DB.Where("user_id = ?", userID).Order("created_at desc").Find(&files)

		// Get all folders for this user
		var folders []Folder
		DB.Where("user_id = ?", userID).Find(&folders)
		folderMap := make(map[string]string)
		for _, folder := range folders {
			folderMap[folder.ID] = folder.Name
		}

		response := make([]map[string]interface{}, len(files))
		for i, f := range files {
			var folderName interface{} = nil
			if f.FolderID != nil {
				if name, ok := folderMap[*f.FolderID]; ok {
					folderName = name
				}
			}
			response[i] = map[string]interface{}{
				"file_id":     f.ID,
				"filename":    f.Filename,
				"path":        f.Path,
				"size":        f.Size,
				"folder_id":   f.FolderID,
				"folder_name": folderName,
				"created_at":  f.CreatedAt,
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"files": response})
		return
	}

	// Fallback: read from uploads directory
	uploadDir := "./uploads"
	entries, err := os.ReadDir(uploadDir)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"files": []interface{}{}})
		return
	}

	var files []map[string]interface{}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, _ := entry.Info()
		parts := strings.SplitN(entry.Name(), "_", 2)
		fileID := parts[0]
		filename := entry.Name()
		if len(parts) > 1 {
			filename = parts[1]
		}

		files = append(files, map[string]interface{}{
			"file_id":    fileID,
			"filename":   filename,
			"path":       filepath.Join(uploadDir, entry.Name()),
			"size":       info.Size(),
			"created_at": info.ModTime(),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"files": files})
}

func DeleteFileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fileID := r.URL.Query().Get("id")

	if userID == "" || fileID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if DB != nil {
		DB.Where("id = ? AND user_id = ?", fileID, userID).Delete(&UploadedFile{})
		DB.Where("source_file_id = ? AND user_id = ?", fileID, userID).Delete(&FineTunedModel{})
	}

	pattern := "./uploads/" + fileID + "_*"
	matches, _ := filepath.Glob(pattern)
	for _, match := range matches {
		os.Remove(match)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}
