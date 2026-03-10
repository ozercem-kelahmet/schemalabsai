#!/bin/bash
set -e
echo "Deploying to GCP (Docker)..."

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai"

cd ~/Desktop/schemalabsai

# ---- GIT SYNC ----
echo "Git sync..."
git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" ":!terraform" || true
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin main --force || true

# ---- UNLOCK REMOTE ----
echo "Unlocking remote..."
ssh $SERVER 'sudo chattr -i / 2>/dev/null; sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null; sudo chattr -R -i /opt/schemalabsai/frontend/.next 2>/dev/null; sudo rm -rf /opt/schemalabsai/frontend/.next; echo "UNLOCKED"'

# ---- RSYNC ----
echo "Syncing files..."
rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='google_credentials.json' \
  --include='.dockerignore' \
  --include='docker-compose.yml' \
  --include='docker/' --include='docker/***' \
  --include='model/' --include='model/*.py' --include='model/adapters/***' --include='model/layers/***' --include='model/miras/***' --include='model/inference/***' --exclude='model/finetuned_models' --exclude='model/checkpoints' --exclude='model/data' --exclude='model/uploads' \
  --include='handlers/' --include='handlers/*.go' \
  --include='services/' --include='services/*.go' \
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
  ~/Desktop/schemalabsai/ $SERVER:$REMOTE_DIR/

# ---- UPDATE DOCKERFILE.FRONTEND ----
echo "Updating Dockerfile.frontend..."
ssh $SERVER 'cat > /opt/schemalabsai/docker/Dockerfile.frontend << '\''DF'\''
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget -qO- http://localhost:3000 || exit 1
ENV NODE_ENV=production
ENV PORT=3000
CMD ["node", "server.js"]
DF'

# ---- REMOTE BUILD ----
echo "Running remote build..."
ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=360 -o TCPKeepAlive=yes $SERVER << 'DEPLOY_EOF'
#!/bin/bash
set -e
cd /opt/schemalabsai

echo "====== STEP 1: Unlock ======"
sudo chattr -i / 2>/dev/null || true
sudo chattr -i /opt/schemalabsai/frontend 2>/dev/null || true

echo "====== STEP 2: Malware scan ======"
MALWARE=0
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do
  echo "MALWARE: $f"; sudo rm -f "$f"; MALWARE=1
done
SUSPECT=$(ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | wc -l)
if [ "$SUSPECT" -gt 0 ]; then
  ps aux | awk '$3>80' | grep -v -E 'python|node|next|postgres|redis|nginx|sshd|systemd|journalctl|go|schemalabsai|awk|ps|npm|docker' | awk '{print $2}' | xargs -r sudo kill -9
  MALWARE=1
fi
[ "$MALWARE" -eq 0 ] && echo "No malware" || echo "Malware cleaned"

echo "====== STEP 3: Disable old systemd ======"
sudo systemctl stop schemalabsai schemalabs-frontend schemalabsai-flask schemalabs-go 2>/dev/null || true
sudo systemctl disable schemalabsai schemalabs-frontend schemalabsai-flask schemalabs-go 2>/dev/null || true

echo "====== STEP 4: Swap check ======"
SWAP=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP" -lt 1000 ]; then
  sudo swapoff -a 2>/dev/null || true; sudo rm -f /swapfile
  sudo fallocate -l 8G /swapfile; sudo chmod 600 /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile
  echo "Swap 8GB created"
else
  echo "Swap OK (${SWAP}MB)"
fi

echo "====== STEP 5: Docker prep ======"
sudo systemctl start docker 2>/dev/null || true
sudo docker stop schemalabs-flask schemalabs-go schemalabs-frontend 2>/dev/null || true
sudo docker rm -f schemalabs-flask schemalabs-go schemalabs-frontend 2>/dev/null || true
sudo fuser -k 6000/tcp 3000/tcp 8080/tcp 2>/dev/null || true
sleep 3

echo "====== STEP 6: Sync requirements ======"
/opt/schemalabsai/venv/bin/pip freeze | grep -v -E "^torch|^torchvision|^nvidia|^cuda|^cudf|^cupy" > /opt/schemalabsai/model/requirements.txt
echo "Requirements synced"

echo "====== STEP 7: Docker build (BuildKit + cache) ======"
sudo docker image prune -f 2>/dev/null || true

echo "Memory before build:"
free -h | head -3

build_svc() {
  local SVC=$1
  local LOG="/tmp/build-${SVC}.log"
  echo "--- Building ${SVC} ---"
  sudo DOCKER_BUILDKIT=1 docker compose build --progress=plain ${SVC} 2>&1 | tee ${LOG}
  local EXIT=${PIPESTATUS[0]}
  if [ "$EXIT" -eq 0 ]; then
    echo "${SVC} OK ($(sudo docker images schemalabsai-${SVC} --format '{{.Size}}' 2>/dev/null))"
  else
    echo "${SVC} FAILED:"
    tail -20 ${LOG}
    return 1
  fi
}

build_svc go || exit 1

FLASK_IMG=$(sudo docker images -q schemalabsai-flask 2>/dev/null)
if [ -z "$FLASK_IMG" ]; then
  build_svc flask || exit 1
else
  echo "Flask reused ($(sudo docker images schemalabsai-flask --format '{{.Size}}'))"
fi

build_svc frontend || exit 1

echo "====== STEP 8: Docker up ======"
sudo fuser -k 6000/tcp 3000/tcp 8080/tcp 2>/dev/null || true
sleep 2
sudo docker compose up -d postgres redis flask go frontend
sleep 5

GRAFANA=$(sudo docker ps -q -f name=schemalabs-grafana 2>/dev/null)
if [ -z "$GRAFANA" ]; then
  echo "Starting monitoring..."
  sudo docker compose up -d grafana prometheus node-exporter 2>/dev/null || true
fi

echo "====== STEP 9: Health checks ======"
for i in $(seq 1 10); do
  sudo docker exec schemalabs-postgres pg_isready -U schemalabs > /dev/null 2>&1 && echo "PostgreSQL OK" && break
  echo "Waiting PostgreSQL... ($i/10)"; sleep 3
done

RPWD=$(grep REDIS_PASSWORD /opt/schemalabsai/.env | cut -d= -f2)
for i in $(seq 1 5); do
  sudo docker exec schemalabs-redis redis-cli -a $RPWD ping 2>/dev/null | grep -q PONG && echo "Redis OK" && break
  echo "Waiting Redis... ($i/5)"; sleep 3
done

FLASK_OK=0
for i in $(seq 1 30); do
  curl -s --max-time 3 http://localhost:6000/health 2>/dev/null | grep -q '"status":"ok"' && echo "Flask OK" && FLASK_OK=1 && break
  echo "Waiting Flask... ($i/30)"; sleep 3
done
[ "$FLASK_OK" -eq 0 ] && echo "Flask failed" && sudo docker logs schemalabs-flask --tail 20

GO_OK=0
for i in $(seq 1 15); do
  ss -tlnp 2>/dev/null | grep -q ':8080' && echo "Go OK" && GO_OK=1 && break
  echo "Waiting Go... ($i/15)"; sleep 3
done
[ "$GO_OK" -eq 0 ] && echo "Go failed" && sudo docker logs schemalabs-go --tail 20

NEXT_OK=0
for i in $(seq 1 15); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000" 2>/dev/null || echo "000")
  [ "$CODE" = "200" ] || [ "$CODE" = "307" ] || [ "$CODE" = "302" ] && echo "Next.js OK (HTTP $CODE)" && NEXT_OK=1 && break
  echo "Waiting Next.js... ($i/15)"; sleep 3
done

echo "====== STEP 10: Lock & security ======"
sudo chattr +i / 2>/dev/null || true
sudo chattr +i /opt/schemalabsai/frontend 2>/dev/null || true
sudo systemctl restart schemalabs-website.service 2>/dev/null || true

CLEAN=1
for f in $(sudo find / -maxdepth 1 -type f -executable 2>/dev/null); do echo "MALWARE: $f"; CLEAN=0; done
[ "$CLEAN" -eq 1 ] && echo "No malware"

echo ""
echo "========== FINAL STATUS =========="
echo "Containers:"
sudo docker ps --format "  {{.Names}}: {{.Status}}"
echo ""
echo "Flask: $(curl -s --max-time 3 http://localhost:6000/health 2>/dev/null || echo FAILED)"
echo "Go: $(curl -s --max-time 3 http://localhost:8080/api/health 2>/dev/null || echo FAILED)"
echo "Next.js: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:3000 2>/dev/null || echo 000)"
echo "Site: $(curl -sf -o /dev/null -w '%{http_code}' https://console.schemalabs.ai 2>/dev/null || echo 000)"
echo ""
echo "Deploy complete - https://console.schemalabs.ai"
DEPLOY_EOF

echo ""
echo "Deploy complete - https://console.schemalabs.ai"