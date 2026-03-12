#!/bin/bash
LOCAL_DIR="$HOME/Downloads/schemalabs-backups"
SERVER="ozercemkelahmet@34.9.180.204"
mkdir -p $LOCAL_DIR

echo "[$(date)] Downloading latest backups..."

LATEST_PG=$(ssh $SERVER "ls -t /opt/schemalabsai/backups/postgres_*.sql.gz 2>/dev/null | head -1")
LATEST_RDB=$(ssh $SERVER "ls -t /opt/schemalabsai/backups/redis_*.rdb 2>/dev/null | head -1")

if [ -n "$LATEST_PG" ]; then
  scp $SERVER:$LATEST_PG $LOCAL_DIR/
  echo "PostgreSQL: $(basename $LATEST_PG)"
fi

if [ -n "$LATEST_RDB" ]; then
  scp $SERVER:$LATEST_RDB $LOCAL_DIR/
  echo "Redis: $(basename $LATEST_RDB)"
fi

find $LOCAL_DIR -name 'postgres_*.sql.gz' -mtime +3 -delete
find $LOCAL_DIR -name 'redis_*.rdb' -mtime +3 -delete

echo "[$(date)] Done"
ls -lh $LOCAL_DIR
