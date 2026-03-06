package handlers

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// HTTP
	HttpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_http_requests_total",
		Help: "Total HTTP requests by method, path, status",
	}, []string{"method", "path", "status"})

	HttpRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "schemalabs_http_request_duration_seconds",
		Help:    "HTTP request duration in seconds",
		Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
	}, []string{"method", "path"})

	HttpErrorsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_http_errors_total",
		Help: "Total HTTP 4xx/5xx errors",
	}, []string{"method", "path", "status"})

	// Training
	TrainingJobsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_training_jobs_total",
		Help: "Total training jobs by status (started/completed/failed)",
	}, []string{"status"})

	TrainingJobsActive = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "schemalabs_training_jobs_active",
		Help: "Currently active training jobs",
	})

	TrainingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "schemalabs_training_duration_seconds",
		Help:    "Training job duration in seconds",
		Buckets: []float64{10, 30, 60, 120, 300, 600, 1200, 3600},
	})

	// Uploads
	UploadsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_uploads_total",
		Help: "Total file uploads by type (csv/excel/json)",
	}, []string{"type"})

	UploadBytesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_upload_bytes_total",
		Help: "Total bytes uploaded by type",
	}, []string{"type"})

	UploadErrorsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_upload_errors_total",
		Help: "Total upload errors by reason",
	}, []string{"reason"})

	// Inference / Predict
	InferenceRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_inference_requests_total",
		Help: "Total inference requests by status",
	}, []string{"status"})

	InferenceDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "schemalabs_inference_duration_seconds",
		Help:    "Inference request duration in seconds",
		Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
	})

	// Auth / Users
	AuthEventsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_auth_events_total",
		Help: "Auth events: login/logout/register/failed",
	}, []string{"event"})

	ActiveUsersGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "schemalabs_active_users",
		Help: "Currently active user sessions",
	})

	// Chat / LLM
	ChatRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "schemalabs_chat_requests_total",
		Help: "Total chat requests by model",
	}, []string{"model", "status"})

	ChatDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "schemalabs_chat_duration_seconds",
		Help:    "Chat response duration in seconds",
		Buckets: []float64{0.5, 1, 2, 5, 10, 30, 60},
	}, []string{"model"})

	// Models
	ModelsTotal = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "schemalabs_models_total",
		Help: "Total trained models in the system",
	})
)

// MetricsHandler exposes /metrics endpoint
func MetricsHandler(w http.ResponseWriter, r *http.Request) {
	promhttp.Handler().ServeHTTP(w, r)
}

// InstrumentHandler wraps HTTP handlers to record metrics automatically
func InstrumentHandler(path string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, statusCode: 200}
		next(rw, r)
		duration := time.Since(start).Seconds()
		status := strconv.Itoa(rw.statusCode)

		HttpRequestsTotal.WithLabelValues(r.Method, path, status).Inc()
		HttpRequestDuration.WithLabelValues(r.Method, path).Observe(duration)

		if rw.statusCode >= 400 {
			HttpErrorsTotal.WithLabelValues(r.Method, path, status).Inc()
		}
	}
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}
