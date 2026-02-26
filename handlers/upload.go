package handlers

import (
	"encoding/csv"
	"log"
	"encoding/json"
	"github.com/xuri/excelize/v2"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)


// sanitizeFilename - Path traversal önlemek için filename sanitize et
func sanitizeFilename(name string) string {
	// Path separator ve tehlikeli karakterleri kaldır
	name = filepath.Base(name)
	name = strings.ReplaceAll(name, "..", "")
	name = strings.ReplaceAll(name, "/", "")
	name = strings.ReplaceAll(name, "\\", "")
	if name == "" || name == "." {
		return "unnamed_file"
	}
	return name
}
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
	ID           string    `gorm:"primaryKey" json:"file_id"`
	Filename     string    `json:"filename"`
	Path         string    `json:"path"`
	Size         int64     `json:"size"`
	UserID       string    `json:"user_id"`
	FolderID     *string   `json:"folder_id"`
	CreatedAt    time.Time `json:"created_at"`
	Columns      string    `json:"columns"`
	RowCount     int       `json:"row_count"`
	UniqueValues string    `json:"unique_values"`
	Vertical     string    `json:"vertical"`
	Source       string    `json:"source"`
	IsMerged     bool      `json:"is_merged"`
}

type UploadResponse struct {
	FileID   string `json:"file_id"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}


func flattenJSON(prefix string, m map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range m {
		key := k
		if prefix != "" {
			key = prefix + "." + k
		}
		switch val := v.(type) {
		case map[string]interface{}:
			for fk, fv := range flattenJSON(key, val) {
				result[fk] = fv
			}
		default:
			result[key] = v
			_ = val
		}
	}
	return result
}

func UploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	var parseWarning string
	folderID := r.FormValue("folder_id")

// Check storage and credit quota before upload
if userID != "" && DB != nil {
	if ok, reason := CheckStorage(userID, float64(r.ContentLength)/(1024*1024)); !ok {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": reason})
		return
	}
	if ok, reason := CheckCredits(userID, 0.01); !ok {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": reason})
		return
	}
}


	file, header, err := r.FormFile("file")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusBadRequest)
json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read file"})
		return
	}
	defer file.Close()

	maxFileSizeMB := getEnvInt("MAX_FILE_SIZE_MB", 50)
if userID != "" && DB != nil {
if quota, err := GetOrCreateQuota(userID); err == nil && quota != nil {
if quota.Plan == "alpha_unlimited" || quota.Plan == "limitless" || quota.Plan == "unlimited" {
maxFileSizeMB = getEnvInt("MAX_FILE_SIZE_MB_UNLIMITED", 100)
}
}
}
maxFileSize := int64(maxFileSizeMB) * 1024 * 1024
	if header.Size > maxFileSize {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusBadRequest)
json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("File too large. Max size: %dMB", maxFileSizeMB)})
		return
	}

	maxTotalMB := getEnvInt("MAX_TOTAL_STORAGE_MB", 1024)
	maxTotalSize := maxTotalMB * 1024 * 1024
	var totalUsed int64
	if userID != "" && DB != nil {
		DB.Model(&UploadedFile{}).Where("user_id = ?", userID).Select("COALESCE(SUM(size), 0)").Scan(&totalUsed)
	}
	if totalUsed + header.Size > maxTotalSize {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusBadRequest)
json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("Storage limit exceeded. Max: %dMB, Used: %dMB", maxTotalMB, totalUsed/(1024*1024))})
		return
	}

	ext := strings.ToLower(filepath.Ext(header.Filename))
	allowed := map[string]bool{".csv": true, ".xlsx": true, ".xls": true, ".json": true, ".parquet": true, ".txt": true, ".pdf": true}
	if !allowed[ext] {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusBadRequest)
json.NewEncoder(w).Encode(map[string]string{"error": "File type not supported. Supported: CSV, Excel (.xlsx, .xls)"})
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

	destFilename := fileID + "_" + sanitizeFilename(finalFilename)
	destPath := filepath.Join(uploadDir, destFilename)

	dest, err := os.Create(destPath)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusInternalServerError)
json.NewEncoder(w).Encode(map[string]string{"error": "Failed to save file"})
		return
	}

	size, err := io.Copy(dest, file)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusInternalServerError)
json.NewEncoder(w).Encode(map[string]string{"error": "Failed to write file"})
		return
	}
	dest.Sync()
	dest.Close() // Close before parsing so Excel/JSON can read the file

// Verify file written correctly
if fi, statErr := os.Stat(destPath); statErr == nil {
log.Printf("File written: %s, disk=%d bytes, copy=%d bytes", destPath, fi.Size(), size)
} else {
log.Printf("File stat error: %v", statErr)
}
	// Save to database if user is logged in
	if userID != "" && DB != nil {
		// Parse CSV to get columns and unique values
		columns := ""
		rowCount := 0
		uniqueValues := ""
		
		if strings.HasSuffix(strings.ToLower(finalFilename), ".csv") {
			if csvFile, err := os.Open(destPath); err == nil {
				reader := csv.NewReader(csvFile)
				if headers, err := reader.Read(); err == nil {
					columns = strings.Join(headers, ",")
					
					// Find target column (last column or 'sector'/'subsector'/'category')
					targetIdx := len(headers) - 1
					for i, h := range headers {
						hl := strings.ToLower(h)
						if hl == "sector" || hl == "subsector" || hl == "category" || hl == "class" || hl == "target" || hl == "label" {
							targetIdx = i
							break
						}
					}
					
					// Read all rows and collect unique target values
					uniqueMap := make(map[string]bool)
					for {
						record, err := reader.Read()
						if err != nil {
							break
						}
						rowCount++
						if targetIdx < len(record) {
							uniqueMap[record[targetIdx]] = true
						}
					}
					
					// Convert unique values to string
					uniqueList := make([]string, 0, len(uniqueMap))
					for k := range uniqueMap {
						uniqueList = append(uniqueList, k)
					}
					uniqueValues = strings.Join(uniqueList, ",")
				}
				csvFile.Close()
			}
		}

		// Parse Excel files
		if strings.HasSuffix(strings.ToLower(finalFilename), ".xlsx") || strings.HasSuffix(strings.ToLower(finalFilename), ".xls") {
			if xlFile, err := excelize.OpenFile(destPath); err == nil {
				sheets := xlFile.GetSheetList()
log.Printf("Excel opened: %s, sheets=%v (%d)", destPath, sheets, len(sheets))
				if len(sheets) > 0 {
					rows, err := xlFile.GetRows(sheets[0])
log.Printf("Excel sheet[0] %q: rows=%d err=%v", sheets[0], len(rows), err)
					if err == nil && len(rows) > 0 {
						columns = strings.Join(rows[0], ",")
						rowCount = len(rows) - 1
						targetIdx := len(rows[0]) - 1
						for i, h := range rows[0] {
							hl := strings.ToLower(h)
							if hl == "sector" || hl == "subsector" || hl == "category" || hl == "class" || hl == "target" || hl == "label" {
								targetIdx = i
								break
							}
						}
						uniqueMap := make(map[string]bool)
						for i := 1; i < len(rows); i++ {
							if targetIdx < len(rows[i]) {
								uniqueMap[rows[i][targetIdx]] = true
							}
						}
						uniqueList := make([]string, 0, len(uniqueMap))
						for k := range uniqueMap {
							uniqueList = append(uniqueList, k)
						}
						uniqueValues = strings.Join(uniqueList, ",")
					}
				}

// Create connection for multi-sheet Excel files
if len(sheets) >= 1 {
baseName := strings.TrimSuffix(strings.TrimSuffix(finalFilename, ".xlsx"), ".xls")
connID := uuid.New().String()[:16]
var tableDetails []map[string]interface{}
var selectedTableNames []string

// Export ALL sheets (including first) as CSV files under connection
for si := 0; si < len(sheets); si++ {
sheetRows, serr := xlFile.GetRows(sheets[si])
if serr != nil || len(sheetRows) < 1 { continue }
sheetFileID := fmt.Sprintf("conn_%s_%s", connID, sanitizeFilename(sheets[si]))
sheetFilename := fmt.Sprintf("%s - %s.csv", baseName, sanitizeFilename(sheets[si]))
sheetPath := fmt.Sprintf("./uploads/%s.csv", sheetFileID)
sheetFile, ferr := os.Create(sheetPath)
if ferr != nil { continue }
sheetWriter := csv.NewWriter(sheetFile)
for _, row := range sheetRows {
sheetWriter.Write(row)
}
sheetWriter.Flush()
sheetInfo, _ := sheetFile.Stat()
sheetSize := sheetInfo.Size()
sheetFile.Close()
sheetCols := ""
sheetRowCount := len(sheetRows) - 1
sheetColCount := 0
if len(sheetRows) > 0 { sheetCols = strings.Join(sheetRows[0], ","); sheetColCount = len(sheetRows[0]) }
DB.Create(&UploadedFile{
ID: sheetFileID, Filename: sheetFilename, Path: sheetPath,
Size: sheetSize, UserID: userID, CreatedAt: time.Now(),
Columns: sheetCols, RowCount: sheetRowCount, Source: "connection",
})
tableDetails = append(tableDetails, map[string]interface{}{
"name": sheets[si], "rows": sheetRowCount, "columns": sheetColCount,
})
selectedTableNames = append(selectedTableNames, sheets[si])
log.Printf("Excel sheet %d/%d exported: %s (%d rows)", si+1, len(sheets), sheetFilename, sheetRowCount)
}

// Create connection record
if len(tableDetails) > 0 {
cachedJSON, _ := json.Marshal(map[string]interface{}{"table_details": tableDetails})
selectedJSON, _ := json.Marshal(selectedTableNames)
now := time.Now()
DB.Create(&Connection{
ID: connID, Name: baseName, Type: "upload", SubType: "excel",
Status: "active", UserID: userID,
CachedTables: string(cachedJSON), SelectedTables: string(selectedJSON),
CachedAt: &now, CreatedAt: now, UpdatedAt: now,
})
log.Printf("Excel connection created: %s with %d sheets", baseName, len(tableDetails))
}
}
				xlFile.Close()
			} else {
				log.Printf("Excel OpenFile failed for %s: %v", destPath, err)
				parseWarning = fmt.Sprintf("Failed to parse Excel file: %v. The file may be corrupted or in an unsupported format.", err)
				// Check if it's actually an Apple Numbers file
				if zipReader, zerr := os.Open(destPath); zerr == nil {
					buf := make([]byte, 512)
					zipReader.Read(buf)
					zipReader.Close()
					if strings.Contains(string(buf), "Data/Preset") || strings.Contains(string(buf), "Index/Document") {
						log.Printf("File %s appears to be Apple Numbers format, not xlsx", destPath)
					parseWarning = "This file appears to be in Apple Numbers format. Please export as .xlsx from Numbers app: File > Export To > Excel"
						// Try to update the record with a note
						if userID != "" && DB != nil {
							DB.Model(&UploadedFile{}).Where("id = ?", fileID).Updates(map[string]interface{}{
								"columns": "⚠️ Apple Numbers format detected. Please export as .xlsx from Numbers: File > Export To > Excel",
								"row_count": 0,
							})
						}
					}
				}
			}
		}
		

// Parse JSON/JSONL files
log.Printf("Checking file extension: %s", strings.ToLower(finalFilename))
if strings.HasSuffix(strings.ToLower(finalFilename), ".json") || strings.HasSuffix(strings.ToLower(finalFilename), ".jsonl") {
	if jsonData, err := os.ReadFile(destPath); err == nil {
		log.Printf("JSON file read: %d bytes from %s", len(jsonData), destPath)
		var records []map[string]interface{}
		if strings.HasSuffix(strings.ToLower(finalFilename), ".jsonl") {
			for _, line := range strings.Split(strings.TrimSpace(string(jsonData)), "\n") {
				line = strings.TrimSpace(line)
				if line == "" { continue }
				var obj map[string]interface{}
				if json.Unmarshal([]byte(line), &obj) == nil { records = append(records, obj) }
			}
		} else {
			if json.Unmarshal(jsonData, &records) != nil {
				var wrapper map[string]interface{}
				if json.Unmarshal(jsonData, &wrapper) == nil {
					// Try to find nested array first
					foundArray := false
					for _, v := range wrapper {
						if arr, ok := v.([]interface{}); ok {
							for _, item := range arr {
								if m, ok := item.(map[string]interface{}); ok { records = append(records, m) }
							}
							foundArray = true
							break
						}
					}
					// If no nested array, flatten nested object into leaf keys
					if !foundArray {
						flat := flattenJSON("", wrapper)
						records = append(records, flat)
						log.Printf("JSON single object flattened: %d keys", len(flat))
					}
				}
			}
		}
		if len(records) > 0 {
			rowCount = len(records)
			keyMap := make(map[string]bool)
			var keys []string
			for _, rec := range records {
				for k := range rec { if !keyMap[k] { keyMap[k] = true; keys = append(keys, k) } }
			}
			columns = strings.Join(keys, ",")
			targetKey := ""
			for _, k := range keys {
				kl := strings.ToLower(k)
				if kl == "sector" || kl == "subsector" || kl == "category" || kl == "class" || kl == "target" || kl == "label" {
					targetKey = k; break
				}
			}
			if targetKey == "" && len(keys) > 0 { targetKey = keys[len(keys)-1] }
			uniqueMap := make(map[string]bool)
			for _, rec := range records {
				if v, ok := rec[targetKey]; ok && v != nil { uniqueMap[fmt.Sprintf("%v", v)] = true }
			}
			uniqueList := make([]string, 0, len(uniqueMap))
			for k := range uniqueMap { uniqueList = append(uniqueList, k) }
			uniqueValues = strings.Join(uniqueList, ",")
		}
	}
}

		// Skip DB record for Excel files that were converted to connections
		isExcelConnection := (strings.HasSuffix(strings.ToLower(finalFilename), ".xlsx") || strings.HasSuffix(strings.ToLower(finalFilename), ".xls")) && columns != ""
		if isExcelConnection {
			log.Printf("Skipping uploaded_files record for Excel connection: %s", finalFilename)
		} else {
		uploadedFile := UploadedFile{
			ID:           fileID,
			Filename:     finalFilename,
			Path:         destPath,
			Size:         size,
			UserID:       userID,
			CreatedAt:    time.Now(),
			Columns:      columns,
			RowCount:     rowCount,
			UniqueValues: uniqueValues,
			Source:       "upload",
			FolderID:     func() *string { if folderID != "" { return &folderID }; return nil }(),
		}
		DB.Create(&uploadedFile)

}
// Log upload to usage
sizeMB := float64(size) / (1024 * 1024)
storageCost := sizeMB * 0.01
if storageCost < 0.01 { storageCost = 0.01 }
if userID != "" {
var q UserQuota
if DB.Where("user_id = ?", userID).First(&q).Error == nil {
q.CreditsUsed += storageCost
q.StorageUsedMB += sizeMB
DB.Save(&q)
}
DB.Create(&UsageLog{
ID: uuid.New().String(), UserID: userID, EventType: "upload",
EventName: "File Upload", ResourceID: fileID, ResourceName: finalFilename,
CreditsUsed: storageCost, CreatedAt: time.Now(),
})
}
	}

	type SheetInfo struct {
		FileID   string `json:"file_id"`
		Filename string `json:"filename"`
		Size     int64  `json:"size"`
	}
	var sheetFiles []SheetInfo
	if userID != "" && DB != nil {
		var extras []UploadedFile
		DB.Where("user_id = ? AND id != ? AND created_at > ? AND filename LIKE ?",
			userID, fileID, time.Now().Add(-10*time.Second),
			strings.TrimSuffix(strings.TrimSuffix(finalFilename, ".xlsx"), ".xls")+"%").
			Find(&extras)
		for _, ex := range extras {
			sheetFiles = append(sheetFiles, SheetInfo{FileID: ex.ID, Filename: ex.Filename, Size: ex.Size})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"file_id":   fileID,
		"filename":  finalFilename,
		"size":      size,
		"sheets":    sheetFiles,
		"warning":   parseWarning,
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
		DB.Select("id, filename, path, size, folder_id, created_at, columns, row_count, vertical, source, is_merged").Where("user_id = ? AND row_count > 0", userID).Order("created_at desc").Find(&files)

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
			"columns":     f.Columns,
			"row_count":   f.RowCount,
			"vertical":    f.Vertical,
			"source":      f.Source,
"is_merged":   f.IsMerged,
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

	pattern := "./uploads/" + sanitizeFilename(fileID) + "_*"
	matches, _ := filepath.Glob(pattern)
	for _, match := range matches {
		os.Remove(match)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

func UpdateFileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	fileID := r.URL.Query().Get("id")

	if userID == "" || fileID == "" {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	var req struct {
		Filename string `json:"filename"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if DB != nil {
		DB.Model(&UploadedFile{}).Where("id = ? AND user_id = ?", fileID, userID).Update("filename", req.Filename)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "updated", "filename": req.Filename})
}

// Replace GetFileByIDHandler with correct version

func GetFileByIDHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimSuffix(r.URL.Path, "/"), "/")
	fileID := parts[len(parts)-1]
	
	if fileID == "" {
		http.Error(w, "File ID required", http.StatusBadRequest)
		return
	}
	
	// userID already set by AuthMiddleware
	userID := r.Header.Get("X-User-ID")
	
	var upload UploadedFile
	result := DB.Where("(id = ? OR filename = ? OR filename LIKE ?) AND user_id = ?", fileID, fileID, fileID+"%", userID).First(&upload)
	
	if result.Error != nil {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}
	
	fileInfo := map[string]interface{}{
		"file_id":     upload.ID,
		"file_name":   upload.Filename,
		"file_path":   upload.Path,
		"file_size":   upload.Size,
		"uploaded_at": upload.CreatedAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(fileInfo)
}
