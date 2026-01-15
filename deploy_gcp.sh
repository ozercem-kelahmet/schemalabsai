#!/bin/bash
set -e
echo "🚀 Deploying to GCP..."

cd ~/Desktop/schemalabsai

# 1. Git pull, commit ve push
echo "📦 Git sync..."
git pull origin main || true
git add -A
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Servisleri durdur
echo "🛑 Stopping services..."
gcloud compute ssh schemalabsai-prod-gpu001 --zone=us-central1-b --command="
sudo systemctl stop schemalabs-go schemalabs-flask schemalabs-frontend || true
sleep 2
sudo fuser -k 8080/tcp 6000/tcp 3000/tcp 2>/dev/null || true
"

# 3. Core dosyaları yolla
echo "📁 Syncing core files..."
gcloud compute scp main.go schemalabsai-prod-gpu001:/opt/schemalabsai/ --zone=us-central1-b && echo "  ✓ main.go"
gcloud compute scp model/server.py schemalabsai-prod-gpu001:/opt/schemalabsai/model/ --zone=us-central1-b && echo "  ✓ model/server.py"
gcloud compute scp handlers/*.go schemalabsai-prod-gpu001:/opt/schemalabsai/handlers/ --zone=us-central1-b && echo "  ✓ handlers/*.go"
gcloud compute scp frontend/components/*.tsx schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/components/ --zone=us-central1-b && echo "  ✓ frontend/components/*.tsx"
gcloud compute scp frontend/app/*.tsx schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/app/ --zone=us-central1-b && echo "  ✓ frontend/app/*.tsx"
gcloud compute scp frontend/package.json schemalabsai-prod-gpu001:/opt/schemalabsai/frontend/ --zone=us-central1-b && echo "  ✓ frontend/package.json"

# 4. GCP'de build ve restart
echo "🔧 Building and restarting..."
gcloud compute ssh schemalabsai-prod-gpu001 --zone=us-central1-b --command="
cd /opt/schemalabsai

echo 'Building Go...'
/usr/local/go/bin/go build -o schemalabsai

echo 'Installing Python dependencies...'
sudo /opt/schemalabsai/venv/bin/pip install psycopg2-binary -q 2>/dev/null

echo 'Installing npm dependencies...'
cd /opt/schemalabsai/frontend
npm install --silent 2>/dev/null

echo 'Building frontend...'
npm run build

echo 'Starting services...'
sudo systemctl start schemalabs-flask
sleep 2
sudo systemctl start schemalabs-frontend
sleep 2
sudo systemctl start schemalabs-go
sleep 2

echo ''
echo '✅ Services status:'
sudo systemctl status schemalabs-flask schemalabs-frontend schemalabs-go --no-pager
"

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://schemalabs.ai"
