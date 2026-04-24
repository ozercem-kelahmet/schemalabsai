package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"schemalabsai/services"
)

// WebSocket yerine SSE (Server-Sent Events) kullanıyoruz
// SSE daha basit, reverse proxy ile çalışır, fallback polling ile uyumlu

type SSEClient struct {
	ch     chan string
	userID string
	closed bool
}

var (
	sseClients   = make(map[string]*SSEClient)
	sseClientsMu sync.RWMutex
)

func SSEHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}
	queryID := r.URL.Query().Get("query_id")
	if queryID == "" {
		http.Error(w, "query_id required", http.StatusBadRequest)
		return
	}

	// SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	// Unwrap ResponseWriter to get underlying flusher
	type flusherWriter interface {
		http.ResponseWriter
		http.Flusher
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		// Try to unwrap
		type unwrapper interface {
			Unwrap() http.ResponseWriter
		}
		rw := w
		for !ok {
			if uw, isUW := rw.(unwrapper); isUW {
				rw = uw.Unwrap()
				flusher, ok = rw.(http.Flusher)
			} else {
				break
			}
		}
		if !ok {
			http.Error(w, "SSE not supported", http.StatusInternalServerError)
			return
		}
	}

	client := &SSEClient{
		ch:     make(chan string, 50),
		userID: userID,
	}

	key := queryID  // sadece queryID ile key
	sseClientsMu.Lock()
	sseClients[key] = client
	sseClientsMu.Unlock()

	defer func() {
		sseClientsMu.Lock()
		if sseClients[key] == client {
			delete(sseClients, key)
		}
		sseClientsMu.Unlock()
		func() {
			defer func() { recover() }()
			close(client.ch)
		}()
		log.Printf("[SSE] Client disconnected: %s", key)
	}()

	log.Printf("[SSE] Client connected: %s", key)

	// Heartbeat
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case msg, ok := <-client.ch:
			if !ok {
				return
			}
			w.Write([]byte("data: " + msg + "\n\n"))
			flusher.Flush()
		case <-ticker.C:
			w.Write([]byte(": heartbeat\n\n"))
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

// InitSSEKafkaCallback - Kafka consumer'dan gelen event'leri SSE'ye bağla
func InitSSEKafkaCallback() {
	services.OnTrainingProgress = func(userID, queryID string, data map[string]interface{}) {
		BroadcastTrainingProgress(userID, queryID, data)
	}
}

// CloseSSEClient - training tamamlandığında SSE client'ı kapat
func CloseSSEClient(queryID string) {
	sseClientsMu.Lock()
	client, ok := sseClients[queryID]
	if ok {
		delete(sseClients, queryID)
		close(client.ch)
		log.Printf("[SSE] Client closed after training complete: %s", queryID)
	}
	sseClientsMu.Unlock()
}

// BroadcastTrainingProgress - training progress'i SSE client'larına gönder
func BroadcastTrainingProgress(userID, queryID string, data map[string]interface{}) {
	key := queryID  // sadece queryID
	sseClientsMu.RLock()
	client, ok := sseClients[key]
	sseClientsMu.RUnlock()

	if !ok {
		return
	}

	msg, err := json.Marshal(data)
	if err != nil {
		return
	}

	func() {
		defer func() { recover() }()
		select {
		case client.ch <- string(msg):
		default:
			log.Printf("[SSE] Client buffer full: %s", key)
		}
	}()
}
