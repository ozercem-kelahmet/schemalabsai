#!/bin/bash
set -e

TOTAL_STEPS=8
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
echo "SchemaLabs STAGE Deploy"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo ""

SERVER="ozercemkelahmet@34.9.180.204"
REMOTE_DIR="/opt/schemalabsai-stage"
SOURCE_DIR=~/Desktop/schemalabsai

cd $SOURCE_DIR

step 1 "Git Sync (stage branch)"
git pull origin stage 2>/dev/null || git pull origin main || true
git add -A -- ":!uploads" ":!checkpoints" ":!data" ":!*.csv" ":!*.xlsx" ":!*.bak" ":!*.bak2" ":!*.bak3" ":!*.bak4" ":!terraform" || true
git commit -m "Stage deploy $(date '+%Y-%m-%d %H:%M')" || true
git push origin stage 2>/dev/null || git push origin HEAD:stage --force || true
echo "[OK] Git synced (stage)"

step 2 "Rsync Files to Stage"
rsync -avz -e ssh \
  --include='main.go' --include='go.mod' --include='go.sum' \
  --include='google_credentials.json' \
  --include='.dockerignore' \
  --include='docker/' --include='docker/***' \
  --include='model/' --include='model/*.py' --include='model/requirements.txt' --include='model/adapters/***' --include='model/layers/***' --include='model/miras/***' --include='model/inference/***' --exclude='model/finetuned_models' --exclude='model/checkpoints' --exclude='model/data' --exclude='model/uploads' \
  --include='handlers/' --include='handlers/*.go' \
  --include='services/' --include='services/*.go' --include='services/spark_app/***' \
  --include='airflow/' --include='airflow/dags/***' \
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
  $SOURCE_DIR/ $SERVER:$REMOTE_DIR/ 2>&1 | tail -1
echo "[OK] Files synced to stage"

step 3 "Update Dockerfile.frontend"
cat > /tmp/Dockerfile.frontend.stage << 'DF'
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
scp /tmp/Dockerfile.frontend.stage $SERVER:/tmp/Dockerfile.frontend.stage
ssh $SERVER 'sudo cp /tmp/Dockerfile.frontend.stage /opt/schemalabsai-stage/docker/Dockerfile.frontend'
echo "[OK] Dockerfile updated"

step 3.5 "Override frontend/.env for stage"
cat > /tmp/frontend.env.stage <<FENV
DATABASE_URL=postgresql://schemalabs:$(ssh $SERVER "grep ^POSTGRES_PASSWORD= /opt/schemalabsai-stage/.env.stage | cut -d= -f2-")@postgres-stage:5432/schemalabs
NEXTAUTH_SECRET=$(ssh $SERVER "grep ^NEXTAUTH_SECRET= /opt/schemalabsai/frontend/.env 2>/dev/null | cut -d= -f2- || echo gW0pHizNNY7M4aXW/ZnCu7JAZC1tt4JtKnb2ef+PvcI=")
NEXTAUTH_URL=https://stage.schemalabs.ai
FENV
scp /tmp/frontend.env.stage $SERVER:/tmp/frontend.env.stage
ssh $SERVER "sudo cp /tmp/frontend.env.stage /opt/schemalabsai-stage/frontend/.env"
echo "[OK] frontend/.env stage-specific"

step 4 "Remote Build & Deploy (Stage)"
ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=360 -o TCPKeepAlive=yes $SERVER << 'DEPLOY_EOF'
#!/bin/bash
cd /opt/schemalabsai-stage

step() {
  local N=$1; shift
  local PCT=$((N * 100 / 8))
  local FILLED=$((PCT / 5))
  local EMPTY=$((20 - FILLED))
  local BAR=$(printf "%${FILLED}s" | tr ' ' '#')
  local SPC=$(printf "%${EMPTY}s" | tr ' ' '-')
  echo ""
  echo "=== [${BAR}${SPC}] ${PCT}% === STEP ${N}/8: $1"
}

step 4 "Stop Stage Containers"
sudo docker stop schemalabs-flask-stage schemalabs-go-stage schemalabs-frontend-stage 2>/dev/null || true
sudo docker rm -f schemalabs-flask-stage schemalabs-go-stage schemalabs-frontend-stage 2>/dev/null || true
sudo fuser -k 6001/tcp 3100/tcp 8090/tcp 2>/dev/null || true
sleep 2
echo "[OK] Stage containers stopped"

step 5 "Docker Build (Stage)"
sudo docker image prune -f > /dev/null 2>&1 || true

build_svc() {
  local SVC=$1
  local BK=$2
  local LOG="/tmp/build-stage-${SVC}.log"
  local START=$(date +%s)
  echo "Building ${SVC} (BuildKit=$BK)..."
  local PROGRESS=""
  [ "$BK" -eq 1 ] && PROGRESS="--progress=plain"

  sudo DOCKER_BUILDKIT=$BK docker compose --env-file .env.stage -f docker-compose.stage.yml build $PROGRESS ${SVC} > ${LOG} 2>&1
  local EXIT=$?
  cat ${LOG}

  local DUR=$(( $(date +%s) - START ))
  if [ "$EXIT" -eq 0 ]; then
    echo "[OK] ${SVC} (${DUR}s)"
    return 0
  else
    echo "[FAIL] ${SVC} after ${DUR}s"
    return 1
  fi
}

build_svc go-stage 0 || exit 1
build_svc flask-stage 0 || exit 1
build_svc frontend-stage 1 || exit 1
build_svc spark-app-stage 0 || { echo "[WARN] Spark build failed, continuing"; }

step 6 "Start Stage Containers"
sudo fuser -k 6001/tcp 3100/tcp 8090/tcp 2>/dev/null || true
sleep 2
sudo docker compose --env-file .env.stage -f docker-compose.stage.yml up -d postgres-stage redis-stage flask-stage go-stage frontend-stage
sudo docker compose --env-file .env.stage -f docker-compose.stage.yml up -d spark-app-stage 2>/dev/null || echo "[WARN] Spark failed"
sleep 5
sudo docker cp /opt/schemalabsai-stage/model/server.py schemalabs-flask-stage:/app/model/server.py 2>/dev/null || true
sudo docker restart schemalabs-flask-stage 2>/dev/null || true
echo "[OK] Flask server.py updated"
sudo docker compose --env-file .env.stage -f docker-compose.stage.yml up -d zookeeper-stage 2>/dev/null || true
sleep 10
sudo docker compose --env-file .env.stage -f docker-compose.stage.yml up -d kafka-stage 2>/dev/null || true
sleep 10
sudo docker compose --env-file .env.stage -f docker-compose.stage.yml up -d airflow-stage 2>/dev/null || true
sleep 20
sudo docker exec schemalabs-airflow-stage airflow db upgrade 2>/dev/null || true
echo "[OK] Stage containers started"

step 7 "Health Checks"
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

check_service "PostgreSQL Stage" "sudo docker exec schemalabs-postgres-stage pg_isready -U schemalabs" 10 3
RPWD=$(grep REDIS_PASSWORD /opt/schemalabsai-stage/.env.stage | cut -d= -f2)
check_service "Redis Stage" "sudo docker exec schemalabs-redis-stage redis-cli -a $RPWD ping 2>/dev/null | grep -q PONG" 5 3
check_service "Flask Stage" "curl -s --max-time 3 http://localhost:6001/health | grep -q ok" 30 3 || sudo docker logs schemalabs-flask-stage --tail 20
check_service "Go Stage" "ss -tlnp | grep -q ':8090'" 15 3 || sudo docker logs schemalabs-go-stage --tail 20
check_service "Frontend Stage" "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:3100 | grep -qE '200|302|307'" 20 3

step 8 "Status"
echo ""
echo "============================================================"
echo "  STAGE DEPLOY STATUS"
echo "============================================================"
echo ""
echo "  Containers:"
sudo docker ps --filter "name=-stage" --format "    {{.Names}}: {{.Status}}"
echo ""
echo "  Flask:     $(curl -s --max-time 3 http://localhost:6001/health 2>/dev/null || echo FAILED)"
echo "  Go:        $(curl -s --max-time 3 http://localhost:8090/api/health 2>/dev/null || echo FAILED)"
echo "  Frontend:  HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:3100 2>/dev/null || echo 000)"
echo "  Site:      HTTP $(curl -sf -o /dev/null -w '%{http_code}' https://stage.schemalabs.ai 2>/dev/null || echo 000)"
echo ""
echo "  Memory:    $(free -m | awk '/Mem/{printf "%dMB / %dMB (%.0f%%)", $3, $2, $3/$2*100}')"
echo "  Disk:      $(df -h / | awk 'NR==2{printf "%s / %s (%s)", $3, $2, $5}')"
echo ""
echo "============================================================"
echo "  https://stage.schemalabs.ai"
echo "============================================================"
DEPLOY_EOF

DEPLOY_END=$(date +%s)
DEPLOY_DUR=$(( DEPLOY_END - DEPLOY_START ))
echo ""
echo "Total stage deploy time: $((DEPLOY_DUR / 60))m $((DEPLOY_DUR % 60))s"
echo ""
