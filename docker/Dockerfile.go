FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
COPY handlers/ ./handlers/
COPY services/ ./services/
RUN CGO_ENABLED=0 GOOS=linux GOGC=50 go build -p 1 -o schemalabsai main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates curl tzdata
WORKDIR /app
COPY --from=builder /app/schemalabsai .
COPY .env* ./
COPY google_credentials.json ./
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["./schemalabsai"]
