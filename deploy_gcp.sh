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
gcloud compute scp main.go schemalabsai-prod-gpu001:/opt/schemalabsai/ --zone=us-central1-b
gcloud compute scp model/server.py schemalabsai-prod-gpu001:/opt/schemalabsai/model/ --zone=us-central1-b
gcloud compute scp handlers/*.go schemalabsai-prod-gpu001:/opt/schemalabsai/handlers/ --zone=us-central1-b
gcloud compute scp frontend/components/*.tsx schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/components/ --zone=us-central1-b
gcloud compute scp frontend/app/*.tsx schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/app/ --zone=us-central1-b
gcloud compute scp frontend/package.json schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b

# 3. Build ve restart - tek SSH bağlantısı
echo "🔧 Building and restarting..."
gcloud compute ssh schemalabsai-prod-gpu001 --zone=us-central1-b -- bash -c '
cd /opt/schemalabsai
sudo systemctl stop schemalabs-go schemalabs-flask schemalabs-frontend 2>/dev/null || true
sleep 1
sudo pkill -9 -f "next-server" 2>/dev/null || true
sudo pkill -9 -f "schemalabsai" 2>/dev/null || true
sudo pkill -9 -f "server.py" 2>/dev/null || true
sleep 2

echo "Building Go..."
/usr/local/go/bin/go build -o schemalabsai

echo "Building frontend..."
cd /opt/schemalabsai/frontend
npm install --silent 2>/dev/null || true
npm run build

echo "Starting services..."
sudo systemctl start schemalabs-flask
sleep 2
sudo systemctl start schemalabs-frontend
sleep 2
sudo systemctl start schemalabs-go
sleep 2

echo "Status:"
sudo systemctl status schemalabs-flask schemalabs-frontend schemalabs-go --no-pager
'

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://schemalabs.ai"
