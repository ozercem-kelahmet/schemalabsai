#!/bin/bash
source ~/Desktop/schemalabsai/.env
PG_PASS=$(echo "$DATABASE_URL" | sed 's|.*://[^:]*:\([^@]*\)@.*|\1|')
kubectl create namespace schemalabs 2>/dev/null || true
kubectl delete secret schemalabs-secrets -n schemalabs 2>/dev/null || true
kubectl create secret generic schemalabs-secrets -n schemalabs \
  --from-literal=DATABASE_URL="$DATABASE_URL" \
  --from-literal=REDIS_URL="$REDIS_URL" \
  --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
  --from-literal=POSTGRES_PASSWORD="$PG_PASS"
echo "✅ Secrets created from .env"
