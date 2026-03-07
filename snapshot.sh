#!/bin/bash
set -e

DATE=$(date +%Y%m%d-%H%M%S)
SNAPSHOT_NAME="schemalabsai-prod-$DATE"
DISK="schemalabsai-prod"
ZONE="us-central1-b"
PROJECT="schema-478207"
LOCAL_DIR="$HOME/Downloads/schemalabs-backups"
mkdir -p $LOCAL_DIR

echo "[$DATE] Creating snapshot: $SNAPSHOT_NAME"

gcloud compute snapshots create $SNAPSHOT_NAME \
  --source-disk=$DISK \
  --source-disk-zone=$ZONE \
  --project=$PROJECT \
  --quiet

echo "Snapshot created: $SNAPSHOT_NAME"

# Cleanup snapshots older than 2 days
gcloud compute snapshots list \
  --project=$PROJECT \
  --filter="name~'schemalabsai-prod-' AND creationTimestamp<$(date -v-2d +%Y-%m-%dT00:00:00)" \
  --format="value(name)" | while read snap; do
    echo "Deleting old snapshot: $snap"
    gcloud compute snapshots delete $snap --project=$PROJECT --quiet
done

# Download latest postgres and redis backups
echo "Downloading backups from GCP..."
LATEST_PG=$(ssh ozercemkelahmet@34.9.180.204 "ls -t /opt/schemalabsai/backups/postgres_*.sql.gz | head -1")
LATEST_RDB=$(ssh ozercemkelahmet@34.9.180.204 "ls -t /opt/schemalabsai/backups/redis_*.rdb | head -1")

scp ozercemkelahmet@34.9.180.204:$LATEST_PG $LOCAL_DIR/
scp ozercemkelahmet@34.9.180.204:$LATEST_RDB $LOCAL_DIR/

# Cleanup local backups older than 3 days
find $LOCAL_DIR -name 'postgres_*.sql.gz' -mtime +3 -delete
find $LOCAL_DIR -name 'redis_*.rdb' -mtime +3 -delete

echo "[$DATE] All done. Files in $LOCAL_DIR"
ls -lh $LOCAL_DIR
