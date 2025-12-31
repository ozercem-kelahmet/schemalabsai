package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type ModelClient struct {
	pythonServerURL string
}

func NewModelClient() *ModelClient {
	return &ModelClient{
		pythonServerURL: "http://localhost:6000",
	}
}

func (mc *ModelClient) Train(fileID string, sector string) (map[string]interface{}, error) {
	payload := map[string]string{
		"file_id": fileID,
		"sector":  sector,
	}

	jsonData, _ := json.Marshal(payload)

	resp, err := http.Post(
		mc.pythonServerURL+"/train",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	return result, nil
}

func (mc *ModelClient) Health() error {
	resp, err := http.Get(mc.pythonServerURL + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unhealthy: %d", resp.StatusCode)
	}

	return nil
}
