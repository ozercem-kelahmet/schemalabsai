#!/bin/bash
cd ~/Desktop/schemalabsai

# Portları temizle
lsof -ti:8080,3000,6000 | xargs kill -9 2>/dev/null || true

# Build ve çalıştır
docker-compose up --build
