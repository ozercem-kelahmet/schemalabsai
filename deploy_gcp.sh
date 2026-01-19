#!/bin/bash
set -e
echo "🚀 Deploying to GCP..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

# 1. Git sync
echo "📦 Git sync..."
git pull origin main || true
git add -A
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Dosyaları yolla
echo "📁 Syncing files..."
scp main.go go.mod go.sum $SERVER:$REMOTE_DIR/
scp model/server.py $SERVER:$REMOTE_DIR/model/
scp handlers/*.go $SERVER:$REMOTE_DIR/handlers/
scp -r frontend/components $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/lib $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/hooks $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/app $SERVER:$REMOTE_DIR/frontend/
scp frontend/package.json frontend/tsconfig.json frontend/next.config.mjs $SERVER:$REMOTE_DIR/frontend/

# 3. Remote build ve restart
echo "🔧 Building and restarting..."
ssh $SERVER "
cd $REMOTE_DIR

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
