FROM golang:1.24-alpine AS go-builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o schemalabsai main.go

FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build
RUN rm -rf node_modules && npm install --production

FROM python:3.11-slim
WORKDIR /app

COPY --from=go-builder /app/schemalabsai .
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package*.json ./frontend/
COPY --from=frontend-builder /app/frontend/node_modules ./frontend/node_modules

COPY model/ ./model/
RUN pip install --no-cache-dir flask numpy pandas
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

ENV FLASK_PORT=6000
ENV FRONTEND_PORT=3000
ENV API_PORT=8080

EXPOSE 8080 3000 6000

CMD ["./schemalabsai"]
