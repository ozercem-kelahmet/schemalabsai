# Build stage - Go
FROM golang:1.21-alpine AS go-builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o schemalabsai main.go

# Build stage - Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Final stage
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Go binary
COPY --from=go-builder /app/schemalabsai .

# Copy frontend build
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package*.json ./frontend/

# Copy Python model
COPY model/ ./model/
COPY checkpoints/ ./checkpoints/
RUN pip install --no-cache-dir torch flask numpy pandas

# Environment
ENV FLASK_PORT=6000
ENV FRONTEND_PORT=3000
ENV API_PORT=8080

EXPOSE 8080 3000 6000

CMD ["./schemalabsai"]
