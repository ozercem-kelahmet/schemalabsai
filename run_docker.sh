#!/bin/bash
echo "🐳 Starting Docker..."
cd ~/Desktop/schemalabsai

osascript -e 'quit app "Docker"' 2>/dev/null || true
sleep 3
open /Applications/Docker.app
echo "⏳ Waiting for Docker..."
until docker ps > /dev/null 2>&1; do
  sleep 3
done
echo "✅ Docker ready"

lsof -ti:8080,3000,6000 | xargs kill -9 2>/dev/null || true
docker compose down 2>/dev/null || true
docker compose up --build
