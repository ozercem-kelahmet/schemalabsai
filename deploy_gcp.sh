#!/bin/bash
set -e
echo "🚀 Deploying to GCP..."

cd ~/Desktop/schemalabsai

# 1. Git sync
echo "📦 Git sync..."
git pull origin main || true
git add -A
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Dosyaları yolla
echo "📁 Syncing files..."
gcloud compute scp main.go go.mod go.sum schemalabsai-prod-gpu001:/opt/schemalabsai/ --zone=us-central1-b
gcloud compute scp model/server.py schemalabsai-prod-gpu001:/opt/schemalabsai/model/ --zone=us-central1-b
gcloud compute scp handlers/*.go schemalabsai-prod-gpu001:/opt/schemalabsai/handlers/ --zone=us-central1-b

# Frontend - recursive
gcloud compute scp --recurse frontend/components schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b
gcloud compute scp --recurse frontend/lib schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b
gcloud compute scp --recurse frontend/hooks schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b
gcloud compute scp --recurse frontend/app schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b
gcloud compute scp frontend/package.json frontend/tsconfig.json frontend/next.config.mjs schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b

# 3. Remote build ve restart
echo "🔧 Building and restarting..."
gcloud compute ssh schemalabsai-prod-gpu001 --zone=us-central1-b --command="
cd /opt/schemalabsai

# Stop services
sudo pkill -9 -f '/opt/schemalabsai/schemalabsai' || true
sudo pkill -9 -f 'next-server' || true
sudo pkill -9 -f 'server.py' || true
sleep 2

# Build Go
/usr/local/go/bin/go build -o schemalabsai .

# Build Frontend
cd frontend && npm run build && cd ..

# Start Flask
cd model && nohup /opt/schemalabsai/venv/bin/python -u server.py > /tmp/flask.log 2>&1 &
cd ..
sleep 3

# Start Next.js
cd frontend && nohup npm start > /tmp/next.log 2>&1 &
cd ..
sleep 3

# Start Go
nohup ./schemalabsai > /tmp/app.log 2>&1 &
sleep 5

# Verify
echo '--- Service Status ---'
ss -tlnp | grep -E '8080|3000|6000'
"

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://schemalabs.ai"
