#!/bin/bash
set -e

TOTAL_STEPS=10
step() {
  local N=$1; shift
  local PCT=$((N * 100 / TOTAL_STEPS))
  local FILLED=$((PCT / 5))
  local EMPTY=$((20 - FILLED))
  local BAR=$(printf "%${FILLED}s" | tr ' ' '#')
  local SPC=$(printf "%${EMPTY}s" | tr ' ' '-')
  echo ""
  echo "=== [${BAR}${SPC}] ${PCT}% === STEP ${N}/${TOTAL_STEPS}: $1"
}

DEPLOY_START=$(date +%s)

echo ""
echo "SchemaLabs GCP Deploy"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo ""

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

step 1 "Git Sync"
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" ":!terraform" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main --force || true
echo "[OK] Git synced"

step 2 "Unlock Remote"
ssh $SERVER 'sudo chattr -i / 2>/dev/null; sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null; sudo chattr -R -i /opt/schemalabsai/frontend/.next 2>/dev/null; sudo rm -rf /opt/schemalabsai/frontend/.next' 2>/dev/null
echo "[OK] Remote unlocked"

step 3 "Rsync Files"
rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='google_credentials.json' \
  --include='.dockerignore' \
  --include='docker-compose.yml' \
  --include='docker/' --include='docker/***' --include='docker/Dockerfile.spark' \
  --include='model/' --include='model/*.py' --include='model/adapters/***' --include='model/layers/***' --include='model/miras/***' --include='model/inference/***' --exclude='model/finetuned_models' --exclude='model/checkpoints' --exclude='model/data' --exclude='model/uploads' \
  --include='handlers/' --include='handlers/*.go' \
  --include='services/' --include='services/*.go' --include='services/spark_app/***' \
  --include='frontend/' \
  --include='frontend/.env' --include='frontend/.env.local' --include='frontend/.npmrc' \
  --include='frontend/components/***' --include='frontend/lib/***' \
  --include='frontend/hooks/***' --include='frontend/app/***' \
  --include='frontend/public/***' \
  --include='frontend/package.json' --include='frontend/package-lock.json' --include='frontend/tsconfig.json' \
  --include='frontend/next.config.mjs' --include='frontend/tailwind.config.ts' \
  --include='frontend/postcss.config.mjs' --include='frontend/components.json' \
  --include='frontend/next-env.d.ts' --include='frontend/page.tsx' \
  --exclude='__pycache__' --exclude='*' \
  --rsync-path="sudo rsync" \
  ~/Desktop/schemalabsai/ $SERVER:$REMOTE_DIR/ 2>&1 | tail -1
echo "[OK] Files synced"

step 4 "Update Dockerfile & Compose"
# Dockerfile.frontend: scp ile gonder (zsh heredoc sorun cikarmasin)
cat > /tmp/Dockerfile.frontend << 'DF'
FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
ENV NODE_OPTIONS="--max-old-space-size=2048"
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine
WORKDIR /app
RUN apk --no-cache add curl && addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
DF
scp /tmp/Dockerfile.frontend $SERVER:/tmp/Dockerfile.frontend
ssh $SERVER 'sudo cp /tmp/Dockerfile.frontend /opt/schemalabsai/docker/Dockerfile.frontend'

# Compose fixes (idempotent)
ssh $SERVER 'sudo sed -i "s|wget -qO- http://localhost:3000|curl -sf http://localhost:3000|g" /opt/schemalabsai/docker-compose.yml && grep -q "HOSTNAME=0.0.0.0" /opt/schemalabsai/docker-compose.yml || sudo sed -i "/container_name: schemalabs-frontend/{n;n;n;s/- NODE_ENV=production/- NODE_ENV=production\n      - HOSTNAME=0.0.0.0/}" /opt/schemalabsai/docker-compose.yml'

echo "[OK] Dockerfile & Compose updated"

step 5 "Remote Build & Deploy"
ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=360 -o TCPKeepAlive=yes $SERVER << 'DEPLOY_EOF'
#!/bin/bash
cd /opt/schemalabsai

step() {
  local N=$1; shift
  local PCT=$((N * 100 / 10))
  local FILLED=$((PCT / 5))
  local EMPTY=$((20 - FILLED))
  local BAR=$(printf "%${FILLED}s" | tr ' ' '#')
  local SPC=$(printf "%${EMPTY}s" | tr ' ' '-')
  echo ""
  echo "=== [${BAR}${SPC}] ${PCT}% === STEP ${N}/10: $1"
}

step 5 "Malware Scan"
MALWARE=0
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "[WARN] MALWARE: $f"; sudo rm -f "$f"; MALWARE=1
done
SUSPECT=$(ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | awk '{print $2}' | xargs -r sudo kill -9
  MALWARE=1
fi
[ "$MALWARE" -eq 0 ] && echo "[OK] System clean" || echo "[WARN] Malware cleaned"

step 6 "System Prep"
sudo chattr -i / 2>/dev/null || true
sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null || true
sudo systemctl stop schemalabsai schemalabs-frontend schemalabsai-flask schemalabs-go 2>/dev/null || true
sudo systemctl disable schemalabsai schemalabs-frontend schemalabsai-flask schemalabs-go 2>/dev/null || true

SWAP=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP" -lt 1000 ]; then
  sudo swapoff -a 2>/dev/null || true; sudo rm -f /swapfile
  sudo fallocate -l 8G /swapfile; sudo chmod 600 /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile
  echo "[OK] Swap 8GB created"
else
  echo "[OK] Swap OK (${SWAP}MB)"
fi

sudo systemctl start docker 2>/dev/null || true
sudo docker stop schemalabs-flask schemalabs-go schemalabs-frontend 2>/dev/null || true
sudo docker rm -f schemalabs-flask schemalabs-go schemalabs-frontend 2>/dev/null || true
sudo fuser -k 6000/tcp 3000/tcp 8080/tcp 2>/dev/null || true
sleep 2

/opt/schemalabsai/venv/bin/pip freeze | grep -v -E "^torch|^torchvision|^nvidia|^cuda|^cudf|^cupy" > /opt/schemalabsai/model/requirements.txt
echo "[OK] System ready ($(free -m | awk '/Mem/{print $7}')MB free)"

step 7 "Docker Build"
sudo docker image prune -f > /dev/null 2>&1 || true

build_svc() {
  local SVC=$1
  local BK=$2
  local LOG="/tmp/build-${SVC}.log"
  local START=$(date +%s)
  echo "Building ${SVC} (BuildKit=$BK)..."
  local PROGRESS=""
  [ "$BK" -eq 1 ] && PROGRESS="--progress=plain"

  sudo DOCKER_BUILDKIT=$BK docker compose build $PROGRESS ${SVC} > ${LOG} 2>&1
  local EXIT=$?
  cat ${LOG}

  local DUR=$(( $(date +%s) - START ))
  if [ "$EXIT" -eq 0 ]; then
    echo "[OK] ${SVC} ($(sudo docker images schemalabsai-${SVC} --format '{{.Size}}' 2>/dev/null), ${DUR}s)"
    return 0
  else
    echo "[FAIL] ${SVC} after ${DUR}s"
    return 1
  fi
}

# Go: BUILDKIT=0 (BuildKit kills go compiler with signal 137)
# Flask: BUILDKIT=0 (same reason, large pip install)
# Frontend: BUILDKIT=1 (npm ci layer cache needed)
build_svc go 0 || exit 1

FLASK_IMG=$(sudo docker images -q schemalabsai-flask 2>/dev/null)
if [ -z "$FLASK_IMG" ]; then
  build_svc flask 0 || exit 1
else
  echo "[OK] Flask reused ($(sudo docker images schemalabsai-flask --format '{{.Size}}'))"
fi

build_svc frontend 1 || exit 1

# Spark app build
SPARK_IMG=$(sudo docker images -q schemalabsai-spark-app 2>/dev/null)
if [ -z "$SPARK_IMG" ]; then
  build_svc spark-app 0 || echo "[WARN] Spark build failed, continuing without Spark"
else
  echo "[OK] Spark-app reused ($(sudo docker images schemalabsai-spark-app --format '{{.Size}}'))"
fi

step 8 "Start Containers"
sudo fuser -k 6000/tcp 3000/tcp 8080/tcp 2>/dev/null || true
sleep 2
sudo docker compose up -d postgres redis flask go frontend spark spark-app
sleep 3

GRAFANA=$(sudo docker ps -q -f name=schemalabs-grafana 2>/dev/null)
if [ -z "$GRAFANA" ]; then
  sudo docker compose up -d grafana prometheus node-exporter 2>/dev/null || true
fi
echo "[OK] Containers started"

step 9 "Health Checks"

check_service() {
  local NAME=$1 CMD=$2 RETRIES=$3 DELAY=$4
  for i in $(seq 1 $RETRIES); do
    if eval "$CMD" > /dev/null 2>&1; then
      echo "[OK] $NAME"
      return 0
    fi
    echo "  Waiting $NAME... ($i/$RETRIES)"
    sleep $DELAY
  done
  echo "[FAIL] $NAME"
  return 1
}

check_service "PostgreSQL" "sudo docker exec schemalabs-postgres pg_isready -U schemalabs" 10 3

RPWD=$(grep REDIS_PASSWORD /opt/schemalabsai/.env | cut -d= -f2)
check_service "Redis" "sudo docker exec schemalabs-redis redis-cli -a $RPWD ping 2>/dev/null | grep -q PONG" 5 3

check_service "Flask" "curl -s --max-time 3 http://localhost:6000/health | grep -q '\"status\":\"ok\"'" 30 3 || sudo docker logs schemalabs-flask --tail 20

check_service "Go" "ss -tlnp | grep -q ':8080'" 15 3 || sudo docker logs schemalabs-go --tail 20

check_service "Next.js (HTTP)" "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:3000 | grep -qE '200|302|307'" 20 3

# Docker healthcheck: curl inside container
echo "  Waiting frontend Docker healthcheck..."
for i in $(seq 1 20); do
  STATUS=$(sudo docker inspect --format='{{.State.Health.Status}}' schemalabs-frontend 2>/dev/null || echo "unknown")
  if [ "$STATUS" = "healthy" ]; then
    echo "[OK] Frontend container healthy"
    break
  fi
  if [ "$i" -eq 20 ]; then
    echo "[WARN] Frontend healthcheck timeout - checking manually..."
    sudo docker exec schemalabs-frontend curl -sf http://localhost:3000 > /dev/null 2>&1 && echo "[OK] Frontend responding (healthcheck may be misconfigured)" || echo "[FAIL] Frontend not responding"
  fi
  echo "  Frontend: $STATUS ($i/20)"
  sleep 5
done

step 10 "Lock & Verify"
sudo chattr +i / 2>/dev/null || true
sudo chattr +i /opt/schemalabsai/frontend 2>/dev/null || true
sudo systemctl restart schemalabs-website.service 2>/dev/null || true

CLEAN=1
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do echo "[WARN] MALWARE: $f"; CLEAN=0; done
[ "$CLEAN" -eq 1 ] && echo "[OK] Post-deploy scan clean"

echo ""
echo "============================================================"
echo "  DEPLOY STATUS"
echo "============================================================"
echo ""
echo "  Containers:"
sudo docker ps --format "    {{.Names}}: {{.Status}}"
echo ""
echo "  Flask:   $(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo FAILED)"
echo "  Go:      $(curl -s --max-time 3 http://localhost:8080/api/health 2>/dev/null || echo FAILED)"
echo "  Next.js: HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:3000 2>/dev/null || echo 000)"
echo "  Site:    HTTP $(curl -sf -o /dev/null -w '%{http_code}' https://console.schemalabs.ai 2>/dev/null || echo 000)"
echo ""
echo "  Memory:  $(free -m | awk '/Mem/{printf "%dMB / %dMB (%.0f%%)", $3, $2, $3/$2*100}')"
echo "  Disk:    $(df -h / | awk 'NR==2{printf "%s / %s (%s)", $3, $2, $5}')"
echo ""
echo "============================================================"
echo "  https://console.schemalabs.ai"
echo "============================================================"
DEPLOY_EOF

DEPLOY_END=$(date +%s)
DEPLOY_DUR=$(( DEPLOY_END - DEPLOY_START ))
echo ""
echo "Total deploy time: $((DEPLOY_DUR / 60))m $((DEPLOY_DUR % 60))s"
echo ""