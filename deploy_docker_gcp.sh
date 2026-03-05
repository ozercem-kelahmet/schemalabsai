#!/bin/bash
set -e
echo "🚀 Deploying to GCP (Docker)..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

# 1. Git sync
echo "📦 Git sync..."
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Sync files
echo "📁 Syncing files..."
ssh $SERVER 'sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null; sudo chattr -R -i /opt/schemalabsai/frontend/.next 2>/dev/null; sudo rm -rf /opt/schemalabsai/frontend/.next; echo "UNLOCKED"'

rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='.env' --include='google_credentials.json' \
  --include='.dockerignore' \
  --include='docker-compose.yml' \
  --include='docker/' --include='docker/Dockerfile.go' \
  --include='docker/Dockerfile.flask' --include='docker/Dockerfile.frontend' \
  --include='model/' --include='model/server.py' \
  --include='handlers/' --include='handlers/**.go' \
  --include='services/' --include='services/**.go' \
  --include='frontend/' \
  --include='frontend/.env' --include='frontend/.env.local' --include='frontend/.npmrc' \
  --include='frontend/components/***' --include='frontend/lib/***' \
  --include='frontend/hooks/***' --include='frontend/app/***' \
  --include='frontend/public/***' \
  --include='frontend/package.json' --include='frontend/package-lock.json' --include='frontend/tsconfig.json' \
  --include='frontend/next.config.mjs' --include='frontend/tailwind.config.ts' \
  --include='frontend/postcss.config.mjs' --include='frontend/components.json' \
  --include='frontend/next-env.d.ts' \
  --exclude='*' \
  --rsync-path="sudo rsync" \
  ~/Desktop/schemalabsai/ $SERVER:$REMOTE_DIR/

# 3. Remote build and restart
echo "🔧 Building and restarting..."
ssh $SERVER 'bash -s' << 'REMOTE'
set -e
cd /opt/schemalabsai

echo "====== STEP 1: System stability ======"
SWAP=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP" -lt 1000 ]; then
  echo "Creating 8GB swap..."
  sudo swapoff -a 2>/dev/null || true
  sudo rm -f /swapfile
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi
sudo sysctl -w vm.swappiness=30 > /dev/null 2>&1
sudo sysctl -w vm.overcommit_memory=1 > /dev/null 2>&1
echo "✅ System OK"

echo "====== STEP 2: Stop all services ======"
sudo systemctl stop schemalabsai schemalabs-frontend schemalabsai-flask 2>/dev/null || true
sudo docker compose down 2>/dev/null || true
sleep 2

echo "====== STEP 3: Kill ports ======"
for PORT in 3000 6000 8080; do
  for i in 1 2 3 4 5; do
    PID=$(sudo fuser $PORT/tcp 2>/dev/null | tr -d " ")
    if [ -z "$PID" ]; then break; fi
    sudo kill -9 $PID 2>/dev/null || true
    sleep 1
  done
done
echo "✅ Ports free"

echo "====== STEP 4: Malware scan ======"
MALWARE=0
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "⚠️ MALWARE: $f"
  sudo rm -f "$f"
  MALWARE=1
done
SUSPECT=$(ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | awk '{print $2}' | xargs -r sudo kill -9
  MALWARE=1
fi
[ "$MALWARE" -eq 0 ] && echo "✅ No malware" || echo "⚠️ Malware cleaned"

echo "====== STEP 5: PostgreSQL check ======"
PG_OK=0
for i in 1 2 3; do
  if sudo -u postgres psql -c "SELECT 1" > /dev/null 2>&1; then
    echo "✅ PostgreSQL OK"
    PG_OK=1
    break
  fi
  sudo systemctl restart postgresql
  sleep 5
done
[ "$PG_OK" -eq 0 ] && echo "❌ PostgreSQL failed!" && exit 1

echo "====== STEP 5.5: Docker check ======"
sudo systemctl start docker
sudo systemctl is-enabled docker
echo "✅ Docker autostart OK"

echo "====== STEP 6: Docker build ======"
sudo docker compose build
echo "✅ Docker build OK"

echo "====== STEP 7: Docker up ======"
sudo docker compose up -d
sleep 5

echo "====== STEP 8: Health checks ======"
FLASK_OK=0
for i in $(seq 1 30); do
  HEALTH=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "")
  if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✅ Flask OK"
    FLASK_OK=1
    break
  fi
  echo "Waiting for Flask... ($i/30)"
  sleep 3
done

GO_OK=0
for i in $(seq 1 10); do
  if ss -tlnp 2>/dev/null | grep -q ':8080'; then
    echo "✅ Go OK"
    GO_OK=1
    break
  fi
  echo "Waiting for Go... ($i/10)"
  sleep 3
done

NEXT_OK=0
for i in $(seq 1 15); do
  if ss -tlnp 2>/dev/null | grep -q ':3000'; then
    echo "✅ Next.js OK"
    NEXT_OK=1
    break
  fi
  echo "Waiting for Next.js... ($i/15)"
  sleep 3
done

echo "====== STEP 9: Post-deploy malware scan ======"
CLEAN=1
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "❌ MALWARE: $f"
  CLEAN=0
done
[ "$CLEAN" -eq 1 ] && echo "✅ No malware"

echo "====== STEP 10: Lock directories ======"
sudo chattr +i / 2>/dev/null || true
echo "✅ Locked"

echo ""
echo "========== FINAL STATUS =========="
sudo docker ps
echo ""
FLASK_H=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "FAILED")
echo "Flask: $FLASK_H"
GO_H=$(curl -s --max-time 3 http://localhost:8080/api/health 2>/dev/null || echo "FAILED")
echo "Go: $GO_H"
SITE=$(curl -sf -o /dev/null -w "%{http_code}" https://console.schemalabs.ai 2>/dev/null || echo "000")
echo "Site: $SITE"

if [ "$FLASK_OK" -eq 1 ] && [ "$GO_OK" -eq 1 ] && [ "$NEXT_OK" -eq 1 ] && [ "$CLEAN" -eq 1 ]; then
  echo ""
  echo "✅ Deploy successful!"
else
  echo ""
  echo "⚠️ Deploy completed with warnings"
fi
echo "🌐 https://console.schemalabs.ai"
REMOTE

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://console.schemalabs.ai"
