#!/bin/bash
echo "🚀 Deploying to GCP..."

# 1. Dosyaları kopyala
gcloud compute scp ~/Desktop/schemalabsai/model/server.py schemalabsai-prod:/opt/schemalabsai/model/ --zone=us-central1-b
gcloud compute scp ~/Desktop/schemalabsai/handlers/*.go schemalabsai-prod:/opt/schemalabsai/handlers/ --zone=us-central1-b

# 2. Go binary build et (GCP'de)
gcloud compute ssh schemalabsai-prod --zone=us-central1-b --command="cd /opt/schemalabsai && go build -o schemalabsai"

# 3. Servisleri restart et
gcloud compute ssh schemalabsai-prod --zone=us-central1-b --command="
sudo systemctl stop schemalabs-go schemalabs-flask
sudo fuser -k 8080/tcp 2>/dev/null
sudo fuser -k 6000/tcp 2>/dev/null
sudo /opt/schemalabsai/venv/bin/pip install psycopg2-binary -q
sudo systemctl start schemalabs-flask
sleep 2
sudo systemctl start schemalabs-go
"

echo "✅ Deploy complete!"
