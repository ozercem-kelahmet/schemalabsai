#!/bin/bash
LOCAL_DIR="$HOME/Downloads/schemalabs-backups"
SERVER="ozercemkelahmet@34.9.180.204"
PROJECT="schema-478207"
mkdir -p $LOCAL_DIR

DATE=$(date +%Y%m%d-%H%M%S)
SNAPSHOT_NAME="schemalabsai-prod-$DATE"

echo "[$(date)] Creating snapshot..."
gcloud compute snapshots create $SNAPSHOT_NAME \
  --source-disk=schemalabsai-prod \
  --source-disk-zone=us-central1-b \
  --project=$PROJECT --quiet 2>/dev/null && echo "Snapshot: $SNAPSHOT_NAME" || echo "Snapshot failed (auth?)"

gcloud compute snapshots list --project=$PROJECT \
  --filter="name~schemalabsai-prod- AND creationTimestamp<$(date -v-2d +%Y-%m-%dT00:00:00)" \
  --format="value(name)" 2>/dev/null | while read snap; do
    gcloud compute snapshots delete $snap --project=$PROJECT --quiet 2>/dev/null
done

echo "Downloading backups..."
LATEST_PG=$(ssh $SERVER "ls -t /opt/schemalabsai/backups/postgres_*.sql.gz 2>/dev/null | head -1")
LATEST_RDB=$(ssh $SERVER "ls -t /opt/schemalabsai/backups/redis_*.rdb 2>/dev/null | head -1")

[ -n "$LATEST_PG" ] && scp $SERVER:$LATEST_PG $LOCAL_DIR/ && echo "PG: $(basename $LATEST_PG)"
[ -n "$LATEST_RDB" ] && scp $SERVER:$LATEST_RDB $LOCAL_DIR/ && echo "Redis: $(basename $LATEST_RDB)"

find $LOCAL_DIR -name 'postgres_*.sql.gz' -mtime +3 -delete
find $LOCAL_DIR -name 'redis_*.rdb' -mtime +3 -delete

echo "[$(date)] Done"
