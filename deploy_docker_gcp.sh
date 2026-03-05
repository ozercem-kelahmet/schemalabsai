#!/bin/bash
set -e
echo "🚀 Deploying to GCP (Docker)..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

echo "📦 Git sync..."
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

echo "📁 Syncing files..."
ssh $SERVER 'sudo chattr -i / 2>/dev/null; sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null; sudo chattr -R -i /opt/schemalabsai/frontend/.next 2>/dev/null; sudo rm -rf /opt/schemalabsai/frontend/.next; echo "UNLOCKED"'

rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='.env' --include='google_credentials.json' \
  --include='.dockerignore' \
  --include='docker-compose.yml' \
  --include='docker/' --include='docker/***' \
  --include='model/' --include='model/*.py' --include='model/adapters/***' --include='model/layers/***' --include='model/miras/***' --include='model/inference/***' \
  --include='handlers/' --include='handlers/***' \
  --include='services/' --include='services/***' \
  --include='frontend/' \
  --include='frontend/.env' --include='frontend/.env.local' --include='frontend/.npmrc' \
  --include='frontend/components/***' --include='frontend/lib/***' \
  --include='frontend/hooks/***' --include='frontend/app/***' \
  --include='frontend/public/***' \
  --include='frontend/package.json' --include='frontend/package-lock.json' --include='frontend/tsconfig.json' \
  --include='frontend/next.config.mjs' --include='frontend/tailwind.config.ts' \
  --include='frontend/postcss.config.mjs' --include='frontend/components.json' \
  --include='frontend/next-env.d.ts' --include='frontend/page.tsx' \
   \
  --exclude='*' \
  --rsync-path="sudo rsync" \
  ~/Desktop/schemalabsai/ $SERVER:$REMOTE_DIR/

echo "🔧 Building and restarting..."
ssh $SERVER 'bash -s' << 'REMOTE'
set -e
cd /opt/schemalabsai

echo "====== STEP 1: Unlock ======"
sudo chattr -i / 2>/dev/null || true
sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null || true

echo "====== STEP 2: Pre-build malware scan ======"
MALWARE=0
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "⚠️ MALWARE: $f"
  sudo rm -f "$f"
  MALWARE=1
done
for f in $(sudo find /opt/schemalabsai/frontend -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "⚠️ MALWARE: $f"
  sudo rm -f "$f"
  MALWARE=1
done
SUSPECT=$(ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  echo "⚠️ High CPU suspect processes found, killing..."
  ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | awk '{print $2}' | xargs -r sudo kill -9
  MALWARE=1
fi
if [ "$MALWARE" -eq 0 ]; then
  echo "✅ No malware found"
else
  echo "⚠️ Malware cleaned"
fi

echo "====== STEP 3: Disable old systemd services ======"
sudo systemctl stop schemalabsai schemalabs-frontend schemalabsai-flask 2>/dev/null || true
sudo systemctl disable schemalabsai schemalabs-frontend schemalabsai-flask 2>/dev/null || true

echo "====== STEP 4: Ensure PostgreSQL healthy ======"
PG_OK=0
for i in 1 2 3; do
  if sudo -u postgres psql -c "SELECT 1" > /dev/null 2>&1; then
    echo "✅ PostgreSQL OK"
    PG_OK=1
    break
  fi
  echo "PostgreSQL unhealthy, restarting... ($i/3)"
  sudo systemctl restart postgresql
  sleep 5
done
if [ "$PG_OK" -eq 0 ]; then
  echo "❌ PostgreSQL failed!"
  exit 1
fi

echo "====== STEP 5: Swap check ======"
SWAP=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP" -lt 1000 ]; then
  echo "Creating 8GB swap..."
  sudo swapoff -a 2>/dev/null || true
  sudo rm -f /swapfile
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "✅ Swap 8GB created"
else
  echo "✅ Swap OK (${SWAP}MB)"
fi

echo "====== STEP 6: Docker down ======"
sudo systemctl start docker
sudo docker compose down --remove-orphans 2>/dev/null || true
sudo docker rm -f schemalabs-flask schemalabs-go schemalabs-frontend 2>/dev/null || true
sleep 3

for PORT in 3000 6000 8080; do
  PID=$(sudo fuser $PORT/tcp 2>/dev/null | tr -d " ")
  if [ -n "$PID" ]; then
    echo "Port $PORT held by PID $PID, killing..."
    sudo kill -9 $PID 2>/dev/null || true
  fi
done
sleep 5

echo "====== STEP 7: Docker build ======"
sudo docker compose build
echo "✅ Docker build OK"

echo "====== STEP 8: Docker up ======"
sudo docker compose up -d
sleep 5

echo "====== STEP 9: Flask health check ======"
FLASK_OK=0
for i in $(seq 1 30); do
  HEALTH=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "")
  if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✅ Flask OK (attempt $i)"
    FLASK_OK=1
    break
  fi
  if ! sudo docker inspect --format='{{.State.Running}}' schemalabs-flask 2>/dev/null | grep -q true; then
    echo "⚠️ Flask crashed, restarting..."
    sudo docker compose restart flask
  fi
  echo "Waiting for Flask... ($i/30)"
  sleep 3
done
if [ "$FLASK_OK" -eq 0 ]; then
  echo "❌ Flask failed!"
  sudo docker logs schemalabs-flask --tail 20
  exit 1
fi

echo "====== STEP 10: Go health check ======"
GO_OK=0
for i in $(seq 1 15); do
  if ss -tlnp 2>/dev/null | grep -q ':8080'; then
    echo "✅ Go OK (attempt $i)"
    GO_OK=1
    break
  fi
  echo "Waiting for Go... ($i/15)"
  sleep 3
done
if [ "$GO_OK" -eq 0 ]; then
  echo "❌ Go failed!"
  sudo docker logs schemalabs-go --tail 20
  exit 1
fi

echo "====== STEP 11: Next.js health check ======"
NEXT_OK=0
for i in $(seq 1 15); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ] || [ "$CODE" = "307" ] || [ "$CODE" = "302" ]; then
    echo "✅ Next.js OK (HTTP $CODE, attempt $i)"
    NEXT_OK=1
    break
  fi
  echo "Waiting for Next.js... HTTP $CODE ($i/15)"
  sleep 3
done
if [ "$NEXT_OK" -eq 0 ]; then
  echo "⚠️ Next.js check failed"
  sudo docker logs schemalabs-frontend --tail 10
fi

echo "====== STEP 12: Lock directories ======"
sudo chattr +i / 2>/dev/null || true
sudo chattr +i /opt/schemalabsai/frontend 2>/dev/null || true
echo "✅ Directories locked"

echo "====== STEP 13: Post-deploy malware scan ======"
CLEAN=1
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "❌ MALWARE FOUND: $f"
  CLEAN=0
done
SUSPECT=$(ps aux | grep -E "pm2|bun|miner|xmrig" | grep -v grep | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  echo "❌ SUSPECT PROCESSES:"
  ps aux | grep -E "pm2|bun|miner|xmrig" | grep -v grep
  CLEAN=0
fi
if [ "$CLEAN" -eq 1 ]; then
  echo "✅ No malware detected"
fi

echo ""
echo "========== FINAL STATUS =========="
echo "Containers:"
sudo docker ps --format "  {{.Names}}: {{.Status}}"
echo ""
FLASK_H=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "FAILED")
echo "Flask: $FLASK_H"
GO_H=$(curl -s --max-time 3 http://localhost:8080/api/health 2>/dev/null || echo "FAILED")
echo "Go: $GO_H"
NEXT_H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000" 2>/dev/null || echo "000")
echo "Next.js: HTTP $NEXT_H"
SITE=$(curl -sf -o /dev/null -w "%{http_code}" https://console.schemalabs.ai 2>/dev/null || echo "000")
echo "Site: $SITE"

if [ "$FLASK_OK" -eq 1 ] && [ "$GO_OK" -eq 1 ] && [ "$NEXT_OK" -eq 1 ] && [ "$CLEAN" -eq 1 ]; then
  echo ""
  echo "✅ All healthy, no malware - deploy successful!"
else
  echo ""
  echo "⚠️ Deploy completed with warnings"
fi
echo "🌐 https://console.schemalabs.ai"
REMOTE

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://console.schemalabs.ai"
