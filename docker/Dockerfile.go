FROM alpine:latest
RUN apk --no-cache add ca-certificates curl tzdata
WORKDIR /app
COPY schemalabsai-linux ./schemalabsai
COPY .env* ./
COPY google_credentials.json ./
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["./schemalabsai"]
