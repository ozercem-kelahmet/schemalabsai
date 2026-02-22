#!/bin/bash
set -e
echo "🚀 Deploying to GCP..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

# 1. Git sync
echo "📦 Git sync..."
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main || true

# 2. Sync files - single rsync call (fast)
echo "📁 Syncing files..."

# Unlock directories for sync - MUST succeed
ssh $SERVER 'sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null; sudo chattr -i / 2>/dev/null; echo "UNLOCKED"'

# Sync with rsync, use temp dir then sudo move if permission issues
rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='.env' --include='google_credentials.json' \
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
  --include='frontend/next-env.d.ts' --include='frontend/page.tsx' \
  --exclude='*' \
  --rsync-path="sudo rsync" \
  ~/Desktop/schemalabsai/ $SERVER:$REMOTE_DIR/

# 3. Remote build and restart
echo "🔧 Building and restarting..."
ssh $SERVER 'bash -s' << 'REMOTE'
set -e
cd /opt/schemalabsai

echo "====== STEP 0: Unlock directories ======"
sudo chattr -i / 2>/dev/null || true
sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null || true
echo "✅ Directories unlocked"

echo "====== STEP 1: System stability checks ======"
SWAP=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP" -lt 1000 ]; then
  echo "WARNING: Swap is ${SWAP}MB, creating 8GB swap..."
  sudo swapoff -a 2>/dev/null || true
  sudo rm -f /swapfile
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "✅ Swap 8GB created (runtime only)"
else
  echo "✅ Swap OK (${SWAP}MB)"
fi

sudo sysctl -w vm.swappiness=30 > /dev/null 2>&1
sudo sysctl -w vm.overcommit_memory=1 > /dev/null 2>&1
grep -q 'vm.swappiness=30' /etc/sysctl.conf || echo 'vm.swappiness=30' | sudo tee -a /etc/sysctl.conf > /dev/null
grep -q 'vm.overcommit_memory=1' /etc/sysctl.conf || echo 'vm.overcommit_memory=1' | sudo tee -a /etc/sysctl.conf > /dev/null
echo "✅ sysctl OK"

echo "====== STEP 2: Graceful stop all services ======"
sudo systemctl stop schemalabsai schemalabs-frontend schemalabsai-flask 2>/dev/null || true
sleep 3

echo "====== STEP 3: Ensure all processes dead ======"
for PATTERN in '/opt/schemalabsai/schemalabsai' 'next-server' 'next start' 'server.py' '/opt/schemalabsai/venv/bin/python' 'pt_data_worker'; do
  sudo pkill -9 -f "$PATTERN" 2>/dev/null || true
done
sleep 2

for PORT in 3000 6000 8080; do
  for ATTEMPT in 1 2 3 4 5; do
    PID=$(sudo fuser $PORT/tcp 2>/dev/null | tr -d " ")
    if [ -z "$PID" ]; then
      break
    fi
    echo "Port $PORT held by PID $PID, killing (attempt $ATTEMPT)..."
    sudo kill -9 $PID 2>/dev/null || true
    sleep 2
  done
done

echo "====== STEP 4: Verify ALL ports free ======"
for i in $(seq 1 15); do
  BUSY=0
  for PORT in 3000 6000 8080; do
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
      BUSY=1
      PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
      echo "Port $PORT still held by PID $PID, killing..."
      sudo kill -9 $PID 2>/dev/null || true
    fi
  done
  if [ "$BUSY" -eq 0 ]; then
    echo "✅ All ports free"
    break
  fi
  sleep 2
done

for PORT in 3000 6000 8080; do
  if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    echo "❌ FATAL: Port $PORT still in use after 30s. Aborting."
    ss -tlnp | grep ":$PORT "
    exit 1
  fi
done

echo "====== STEP 5: Pre-build malware scan ======"
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
SUSPECT=$(ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm' | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  echo "⚠️ High CPU suspect processes found, killing..."
  ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm' | awk '{print $2}' | xargs -r sudo kill -9
  MALWARE=1
fi
if [ "$MALWARE" -eq 0 ]; then
  echo "✅ No malware found"
else
  echo "⚠️ Malware cleaned"
fi

echo "====== STEP 6: Ensure PostgreSQL is healthy ======"
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
  echo "❌ PostgreSQL failed! Aborting."
  exit 1
fi

echo "====== STEP 7: Build Go ======"
cd /opt/schemalabsai
/usr/local/go/bin/go build -o schemalabsai . || { echo "❌ Go build failed"; exit 1; }
echo "✅ Go build OK"

echo "====== STEP 8: Build Frontend ======"
cd /opt/schemalabsai/frontend
sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null || true
sudo rm -rf .next/lock .next node_modules/.cache
sudo npm install || true
sudo npm run build || { echo "❌ Frontend build failed"; exit 1; }
BUILD_ID=$(cat .next/BUILD_ID 2>/dev/null || echo "unknown")
CHUNK_COUNT=$(ls .next/static/chunks/ 2>/dev/null | wc -l)
echo "✅ Frontend build OK (BUILD_ID=$BUILD_ID, chunks=$CHUNK_COUNT)"

echo "====== STEP 9: Reset systemd ======"
sudo systemctl reset-failed schemalabsai schemalabsai-flask schemalabs-frontend 2>/dev/null || true
sudo systemctl daemon-reload

echo "====== STEP 10: Start Flask ======"
sudo systemctl start schemalabsai-flask

FLASK_OK=0
for i in $(seq 1 30); do
  HEALTH=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "")
  if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✅ Flask OK (attempt $i)"
    FLASK_OK=1
    break
  fi
  if ! systemctl is-active --quiet schemalabsai-flask; then
    echo "⚠️ Flask crashed, resetting and retrying..."
    sudo fuser -k 6000/tcp 2>/dev/null || true
    sleep 2
    sudo systemctl reset-failed schemalabsai-flask 2>/dev/null || true
    sudo systemctl start schemalabsai-flask
  fi
  echo "Waiting for Flask... ($i/30)"
  sleep 3
done

if [ "$FLASK_OK" -eq 0 ]; then
  echo "❌ Flask failed to start!"
  sudo journalctl -u schemalabsai-flask -n 20 --no-pager
  exit 1
fi

echo "====== STEP 11: Start Next.js ======"
sudo systemctl start schemalabs-frontend

NEXT_OK=0
for i in $(seq 1 15); do
  if ss -tlnp 2>/dev/null | grep -q ':3000'; then
    NEXT_PID=$(ss -tlnp 2>/dev/null | grep ':3000' | grep -oP 'pid=\K[0-9]+' | head -1)
    CHUNK=$(ls /opt/schemalabsai/frontend/.next/static/chunks/ 2>/dev/null | head -1)
    if [ -n "$CHUNK" ]; then
      CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000/_next/static/chunks/$CHUNK" 2>/dev/null || echo "000")
      if [ "$CODE" = "200" ]; then
        echo "✅ Next.js OK (PID=$NEXT_PID, chunks serving correctly)"
        NEXT_OK=1
        break
      else
        echo "Next.js chunks HTTP $CODE, restarting... ($i/15)"
        sudo kill -9 $NEXT_PID 2>/dev/null || true
        sleep 2
        sudo systemctl start schemalabs-frontend
      fi
    fi
  fi
  echo "Waiting for Next.js... ($i/15)"
  sleep 3
done

if [ "$NEXT_OK" -eq 0 ]; then
  echo "⚠️ Next.js chunk test failed, but port may be listening"
  sudo journalctl -u schemalabs-frontend -n 10 --no-pager
fi

echo "====== STEP 12: Start Go ======"
sudo systemctl start schemalabsai

GO_OK=0
for i in $(seq 1 10); do
  if ss -tlnp 2>/dev/null | grep -q ':8080'; then
    echo "✅ Go OK (port 8080 listening, attempt $i)"
    GO_OK=1
    break
  fi
  echo "Waiting for Go... ($i/10)"
  sleep 3
done

if [ "$GO_OK" -eq 0 ]; then
  echo "❌ Go failed to start!"
  sudo journalctl -u schemalabsai -n 20 --no-pager
  exit 1
fi

echo "====== STEP 13: Lock directories ======"
sudo chattr +i / 2>/dev/null || true
sudo chattr +i /opt/schemalabsai/frontend 2>/dev/null || true
echo "✅ Directories locked"

echo "====== STEP 14: Post-deploy malware scan ======"
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
echo "Services:"
for svc in schemalabsai schemalabsai-flask schemalabs-frontend; do
  STATUS=$(sudo systemctl is-active $svc 2>/dev/null || echo "unknown")
  echo "  $svc: $STATUS"
done
echo ""
echo "Ports:"
ss -tlnp | grep -E ':3000|:6000|:8080' || echo "  No ports found!"
echo ""

FLASK_H=$(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo "FAILED")
echo "Flask health: $FLASK_H"

NEXT_PID=$(ss -tlnp 2>/dev/null | grep ':3000' | grep -oP 'pid=\K[0-9]+' | head -1)
CHUNK=$(ls /opt/schemalabsai/frontend/.next/static/chunks/ 2>/dev/null | head -1)
NEXT_H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000/_next/static/chunks/$CHUNK" 2>/dev/null || echo "000")
echo "Next.js: HTTP $NEXT_H (PID=$NEXT_PID)"

GO_H=$(curl -s --max-time 3 http://localhost:8080/api/health 2>/dev/null || echo "FAILED")
echo "Go health: $GO_H"

SITE=$(curl -sf -o /dev/null -w "%{http_code}" https://console.schemalabs.ai 2>/dev/null || echo "000")
echo "Site: $SITE"

if [ "$FLASK_OK" -eq 1 ] && [ "$NEXT_OK" -eq 1 ] && [ "$GO_OK" -eq 1 ] && [ "$CLEAN" -eq 1 ]; then
  echo ""
  echo "✅ All services healthy, no malware - deploy successful!"
else
  echo ""
  echo "⚠️ Deploy completed with warnings"
fi
echo "🌐 https://console.schemalabs.ai"
REMOTE

echo ""
echo "✅ Deploy complete!"
echo "🌐 https://console.schemalabs.ai"