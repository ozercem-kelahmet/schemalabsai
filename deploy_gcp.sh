#!/bin/bash
set -e
echo "🚀 Deploying to GCP..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

# 1. Git sync (uploads hariç)
echo "📦 Git sync..."
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Dosyaları yolla
echo "📁 Syncing files..."
scp main.go go.mod go.sum $SERVER:$REMOTE_DIR/
scp model/server.py $SERVER:$REMOTE_DIR/model/
scp handlers/*.go $SERVER:$REMOTE_DIR/handlers/
scp services/*.go $SERVER:$REMOTE_DIR/services/ 2>/dev/null || true
scp -r frontend/components $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/lib $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/hooks $SERVER:$REMOTE_DIR/frontend/
scp -r frontend/app $SERVER:$REMOTE_DIR/frontend/
scp frontend/package.json frontend/tsconfig.json frontend/next.config.mjs $SERVER:$REMOTE_DIR/frontend/

# 3. Remote build and restart
echo "🔧 Building and restarting..."
ssh $SERVER 'bash -s' << 'REMOTE'
set -e
cd /opt/schemalabsai

# Stop systemd services
echo "Stopping services..."
sudo systemctl stop schemalabsai schemalabsai-next schemalabsai-flask 2>/dev/null || true
sleep 2

# AGGRESSIVE CLEANUP - kill ALL related processes
echo "Killing all related processes..."
sudo pkill -9 -f '/opt/schemalabsai/schemalabsai' 2>/dev/null || true
sudo pkill -9 -f 'next-server' 2>/dev/null || true
sudo pkill -9 -f 'node.*next' 2>/dev/null || true
sudo pkill -9 -f 'npm.*start' 2>/dev/null || true
sudo pkill -9 -f 'server.py' 2>/dev/null || true
sudo pkill -9 -f 'python.*server' 2>/dev/null || true
sudo pkill -9 -f '/opt/schemalabsai/venv/bin/python' 2>/dev/null || true
sleep 3

# Force kill anything on our ports
echo "Clearing ports..."
sudo fuser -k 3000/tcp 2>/dev/null || true
sudo fuser -k 6000/tcp 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 2

# Verify ports are free
echo "Verifying ports are free..."
for i in $(seq 1 15); do
  PORTS_IN_USE=$(ss -tlnp | grep -E ':3000|:6000|:8080' | wc -l)
  if [ "$PORTS_IN_USE" -gt 0 ]; then
    echo "Ports still in use ($PORTS_IN_USE), waiting... ($i/15)"
    sleep 2
  else
    echo "All ports free"
    break
  fi
done

# Build Go
echo "Building Go..."
cd /opt/schemalabsai
/usr/local/go/bin/go build -o schemalabsai .

# Build Frontend
echo "Cleaning .next cache..."
sudo rm -rf /opt/schemalabsai/frontend/.next
echo "Building Frontend..."
cd /opt/schemalabsai/frontend
npm run build
echo "Frontend build complete"

# Reset failed states
sudo systemctl reset-failed schemalabsai schemalabsai-flask schemalabsai-next 2>/dev/null || true

# Start services one by one with verification
echo "Starting Flask..."
sudo systemctl start schemalabsai-flask
sleep 5
for i in $(seq 1 10); do
  if ss -tlnp | grep -q ':6000'; then
    echo "Flask running on 6000"
    break
  fi
  echo "Waiting for Flask... ($i)"
  sleep 2
done

echo "Starting Next.js..."
sudo systemctl start schemalabsai-next
sleep 5
for i in $(seq 1 10); do
  if ss -tlnp | grep -q ':3000'; then
    echo "Next.js running on 3000"
    break
  fi
  echo "Waiting for Next.js... ($i)"
  sleep 2
done

echo "Starting Go..."
sudo systemctl start schemalabsai
sleep 3
for i in $(seq 1 10); do
  if ss -tlnp | grep -q ':8080'; then
    echo "Go running on 8080"
    break
  fi
  echo "Waiting for Go... ($i)"
  sleep 2
done

# Final verification
echo ""
echo "=== FINAL STATUS ==="
sudo systemctl is-active schemalabsai schemalabsai-flask schemalabsai-next
echo ""
echo "=== PORTS ==="
ss -tlnp | grep -E ':3000|:6000|:8080'
echo ""
echo "=== PROCESSES ==="
ps aux | grep -E 'schemalabsai|next-server|server.py' | grep -v grep | wc -l
echo "processes running"
REMOTE

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://console.schemalabs.ai"