package services

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"os"
	"strings"
	"time"

	kafka "github.com/segmentio/kafka-go"
)

type KafkaConsumerService struct {
	Brokers []string
	Topics  []string
	GroupID string
}

var DefaultKafkaConsumer *KafkaConsumerService

// OnTrainingProgress - SSE broadcast callback
var OnTrainingProgress func(userID, queryID string, data map[string]interface{})

func InitKafkaConsumer() {
	brokers := os.Getenv("KAFKA_BOOTSTRAP_SERVERS")
	if brokers == "" {
		brokers = "kafka:9092"
	}

	consumer := &KafkaConsumerService{
		Brokers: strings.Split(brokers, ","),
		Topics:  []string{"training_progress", "training_events"},
		GroupID: "schemalabs-go",
	}

	DefaultKafkaConsumer = consumer
	go consumer.start()
	log.Printf("[KAFKA CONSUMER] Initialized: brokers=%s", brokers)
}

func (k *KafkaConsumerService) isAvailable() bool {
	for _, broker := range k.Brokers {
		conn, err := net.DialTimeout("tcp", strings.TrimSpace(broker), 3*time.Second)
		if err == nil {
			conn.Close()
			return true
		}
	}
	return false
}

func (k *KafkaConsumerService) start() {
	// Kafka hazır olana kadar bekle
	for i := 0; i < 12; i++ {
		if k.isAvailable() {
			break
		}
		log.Printf("[KAFKA CONSUMER] Waiting for Kafka... (%d/12)", i+1)
		time.Sleep(10 * time.Second)
	}

	if !k.isAvailable() {
		log.Printf("[KAFKA CONSUMER] Kafka not available, using Redis fallback")
		return
	}

	log.Printf("[KAFKA CONSUMER] Kafka available, starting consumers")

	for _, topic := range k.Topics {
		go k.consume(topic)
	}
}

func (k *KafkaConsumerService) consume(topic string) {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:        k.Brokers,
		Topic:          topic,
		GroupID:        k.GroupID,
		MinBytes:       1,
		MaxBytes:       1024 * 1024,
		CommitInterval: time.Second,
		StartOffset:    kafka.LastOffset,
	})
	defer reader.Close()

	log.Printf("[KAFKA CONSUMER] Consuming topic: %s", topic)

	for {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		msg, err := reader.ReadMessage(ctx)
		cancel()

		if err != nil {
			if !strings.Contains(err.Error(), "context deadline exceeded") {
				log.Printf("[KAFKA CONSUMER] Error reading %s: %v", topic, err)
			}
			time.Sleep(time.Second)
			continue
		}

		var event map[string]interface{}
		if err := json.Unmarshal(msg.Value, &event); err != nil {
			continue
		}

		// Training progress event → SSE'ye push
		queryID, _ := event["query_id"].(string)
		userID, _ := event["user_id"].(string)

		if queryID != "" && OnTrainingProgress != nil {
			OnTrainingProgress(userID, queryID, event)
			log.Printf("[KAFKA CONSUMER] Pushed to SSE: topic=%s query=%s", topic, queryID)
		}
	}
}
