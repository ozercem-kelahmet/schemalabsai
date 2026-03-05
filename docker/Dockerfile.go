FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
COPY handlers/ ./handlers/
COPY services/ ./services/
RUN CGO_ENABLED=0 GOOS=linux go build -o schemalabsai main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates curl
WORKDIR /app
COPY --from=builder /app/schemalabsai .
COPY .env* ./
COPY google_credentials.json ./
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["./schemalabsai"]
