package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
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
	queryID := r.URL.Query().Get("query_id")
	if userID == "" || queryID == "" {
		http.Error(w, "user_id and query_id required", http.StatusBadRequest)
		return
	}

	// SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	client := &SSEClient{
		ch:     make(chan string, 10),
		userID: userID,
	}

	key := userID + ":" + queryID
	sseClientsMu.Lock()
	sseClients[key] = client
	sseClientsMu.Unlock()

	defer func() {
		sseClientsMu.Lock()
		delete(sseClients, key)
		sseClientsMu.Unlock()
		close(client.ch)
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

// BroadcastTrainingProgress - training progress'i SSE client'larına gönder
func BroadcastTrainingProgress(userID, queryID string, data map[string]interface{}) {
	key := userID + ":" + queryID
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

	select {
	case client.ch <- string(msg):
	default:
		log.Printf("[SSE] Client buffer full: %s", key)
	}
}
