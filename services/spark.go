package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

// SparkService - cloud agnostic Spark job manager
// local mode → standalone → Dataproc geçişi sadece env değişikliği
type SparkService struct {
	Mode       string // local, standalone, dataproc
	MasterURL  string
	AppURL     string // Spark REST API
	HTTPClient *http.Client
}

var DefaultSpark *SparkService

func InitSpark() {
	mode := os.Getenv("SPARK_MODE")
	if mode == "" {
		mode = "local"
	}

	svc := &SparkService{
		Mode: mode,
		HTTPClient: &http.Client{
			Timeout: 300 * time.Second,
		},
	}

	switch mode {
	case "local":
		svc.AppURL = os.Getenv("SPARK_APP_URL")
		if svc.AppURL == "" {
			svc.AppURL = "http://spark:4040"
		}
		svc.MasterURL = "local[4]"
	case "standalone":
		svc.MasterURL = os.Getenv("SPARK_MASTER_URL")
		svc.AppURL = os.Getenv("SPARK_APP_URL")
	case "dataproc":
		svc.MasterURL = os.Getenv("DATAPROC_CLUSTER")
		svc.AppURL = os.Getenv("DATAPROC_URL")
	}

	DefaultSpark = svc
	log.Printf("[SPARK] Initialized: mode=%s master=%s", mode, svc.MasterURL)
}

// SparkJobRequest - connection'dan CSV export job
type SparkJobRequest struct {
	JobType    string            `json:"job_type"`   // export_sql, export_mongo, export_api
	ConnType   string            `json:"conn_type"`  // postgresql, mysql, snowflake, mongodb...
	ConnID     string            `json:"conn_id"`
	OutputPath string            `json:"output_path"`
	Config     map[string]string `json:"config"`
	RowLimit   int64             `json:"row_limit"`
}

type SparkJobResponse struct {
	JobID      string `json:"job_id"`
	Status     string `json:"status"`
	OutputPath string `json:"output_path"`
	RowCount   int64  `json:"row_count"`
	Error      string `json:"error,omitempty"`
}

// ShouldUseSpark - threshold kontrolü
// < 100K satır → Go, > 100K satır → Spark
func (s *SparkService) ShouldUseSpark(estimatedRows int64) bool {
	threshold := int64(100000)
	thresholdEnv := os.Getenv("SPARK_ROW_THRESHOLD")
	if thresholdEnv != "" {
		fmt.Sscanf(thresholdEnv, "%d", &threshold)
	}
	return estimatedRows > threshold
}

// ShouldUseSparkBySize - dosya boyutuna göre
// < 50MB → Go, > 50MB → Spark
func (s *SparkService) ShouldUseSparkBySize(sizeBytes int64) bool {
	thresholdMB := int64(50)
	thresholdEnv := os.Getenv("SPARK_SIZE_THRESHOLD_MB")
	if thresholdEnv != "" {
		fmt.Sscanf(thresholdEnv, "%d", &thresholdMB)
	}
	return sizeBytes > thresholdMB*1024*1024
}

// SubmitJob - Spark'a job gönder
func (s *SparkService) SubmitJob(req SparkJobRequest) (*SparkJobResponse, error) {
	if s.Mode == "local" {
		// Local mode: Spark REST API
		return s.submitLocalJob(req)
	}
	// Standalone/Dataproc: aynı interface, farklı endpoint
	return s.submitLocalJob(req)
}

func (s *SparkService) submitLocalJob(req SparkJobRequest) (*SparkJobResponse, error) {
	body, _ := json.Marshal(req)
	resp, err := s.HTTPClient.Post(
		s.AppURL+"/api/v1/jobs",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("spark job submit failed: %v", err)
	}
	defer resp.Body.Close()

	var jobResp SparkJobResponse
	if err := json.NewDecoder(resp.Body).Decode(&jobResp); err != nil {
		return nil, fmt.Errorf("spark response parse failed: %v", err)
	}
	return &jobResp, nil
}

// WaitForJob - job tamamlanana kadar bekle
func (s *SparkService) WaitForJob(jobID string, timeout time.Duration) (*SparkJobResponse, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := s.HTTPClient.Get(s.AppURL + "/api/v1/jobs/" + jobID)
		if err != nil {
			time.Sleep(5 * time.Second)
			continue
		}
		var jobResp SparkJobResponse
		json.NewDecoder(resp.Body).Decode(&jobResp)
		resp.Body.Close()

		if jobResp.Status == "completed" || jobResp.Status == "failed" {
			return &jobResp, nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return nil, fmt.Errorf("spark job timeout after %v", timeout)
}

// IsAvailable - Spark erişilebilir mi?
func (s *SparkService) IsAvailable() bool {
	if s == nil {
		return false
	}
	resp, err := s.HTTPClient.Get(s.AppURL + "/health")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}
