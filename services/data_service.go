package services

import (
	"encoding/csv"
	"os"
)

type DataService struct {
	uploadDir string
}

func NewDataService() *DataService {
	return &DataService{
		uploadDir: "./uploads",
	}
}

func (ds *DataService) ReadCSV(filepath string) ([][]string, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}

func (ds *DataService) ValidateFile(filepath string) bool {
	_, err := os.Stat(filepath)
	return err == nil
}

func (ds *DataService) GetFileInfo(fileID string) (map[string]interface{}, error) {
	info := make(map[string]interface{})
	info["file_id"] = fileID
	info["status"] = "ready"
	return info, nil
}
