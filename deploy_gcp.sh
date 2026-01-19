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

# 3. Stop services
echo "🛑 Stopping services..."
ssh $SERVER "sudo pkill -9 -f '/opt/schemalabsai/schemalabsai' || true; sudo pkill -9 -f 'next-server' || true; sudo pkill -9 -f 'server.py' || true; sleep 2"

# 4. Build Go
echo "🔨 Building Go..."
ssh $SERVER "cd $REMOTE_DIR && /usr/local/go/bin/go build -o schemalabsai ."

# 5. Build Frontend
echo "🔨 Building Frontend..."
ssh $SERVER "cd $REMOTE_DIR/frontend && npm run build"

# 6. Start Flask
echo "🚀 Starting Flask..."
ssh $SERVER "cd $REMOTE_DIR/model && nohup /opt/schemalabsai/venv/bin/python -u server.py > /tmp/flask.log 2>&1 & sleep 3"

# 7. Start Next.js
echo "🚀 Starting Next.js..."
ssh $SERVER "cd $REMOTE_DIR/frontend && nohup npm start > /tmp/next.log 2>&1 & sleep 3"

# 8. Start Go
echo "🚀 Starting Go server..."
ssh $SERVER "cd $REMOTE_DIR && nohup ./schemalabsai > /tmp/app.log 2>&1 & sleep 5"

# 9. Verify
echo "✅ Verifying..."
ssh $SERVER "ss -tlnp | grep -E '8080|3000|6000'"

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://schemalabs.ai"
