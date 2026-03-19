package services

import (
	"encoding/json"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"
)

type KafkaEvent struct {
	Event     string                 `json:"event"`
	Timestamp string                 `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

type KafkaService struct {
	Brokers   []string
	Available bool
	mu        sync.Mutex
}

var DefaultKafka *KafkaService

func InitKafka() {
	brokers := os.Getenv("KAFKA_BOOTSTRAP_SERVERS")
	if brokers == "" {
		brokers = "kafka:9092"
	}

	svc := &KafkaService{
		Brokers:   strings.Split(brokers, ","),
		Available: false,
	}

	go func() {
		time.Sleep(10 * time.Second)
		svc.Available = svc.testConnection()
		if svc.Available {
			log.Printf("[KAFKA] Connected: %s", brokers)
		} else {
			log.Printf("[KAFKA] Not available, fallback to Redis")
		}
	}()

	DefaultKafka = svc
	log.Printf("[KAFKA] Initialized: brokers=%s", brokers)
}

func (k *KafkaService) testConnection() bool {
	for _, broker := range k.Brokers {
		conn, err := net.DialTimeout("tcp", strings.TrimSpace(broker), 3*time.Second)
		if err == nil {
			conn.Close()
			return true
		}
	}
	return false
}

func (k *KafkaService) Publish(topic string, event KafkaEvent) error {
	if !k.Available {
		log.Printf("[KAFKA] Not available, skipping event: %s", event.Event)
		return nil
	}
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	log.Printf("[KAFKA] Published to %s: %s", topic, string(data))
	return nil
}

func (k *KafkaService) PublishTrainingProgress(queryID string, epoch, epochs int, accuracy, loss float64, status string) {
	if DefaultKafka == nil {
		return
	}
	DefaultKafka.Publish("training_progress", KafkaEvent{
		Event:     "training_progress",
		Timestamp: time.Now().Format(time.RFC3339),
		Data: map[string]interface{}{
			"query_id": queryID,
			"epoch":    epoch,
			"epochs":   epochs,
			"accuracy": accuracy,
			"loss":     loss,
			"status":   status,
		},
	})
}
