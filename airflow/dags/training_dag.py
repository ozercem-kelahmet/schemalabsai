from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import requests
import json
import os

KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
GO_API_URL = os.getenv('GO_API_URL', 'http://go:8080')
FLASK_URL = os.getenv('FLASK_URL', 'http://flask:6000')

default_args = {
    'owner': 'schemalabs',
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': None,
}

# DAG 1: Training monitor + retry
with DAG(
    'training_monitor',
    default_args=default_args,
    description='Monitor training, retry on failure, check accuracy',
    schedule_interval=None,  # Event triggered
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['training'],
) as dag:

    def check_training_status(**context):
        model_id = context['dag_run'].conf.get('model_id')
        query_id = context['dag_run'].conf.get('query_id')
        if not model_id:
            raise ValueError("model_id required")
        
        resp = requests.get(f"{GO_API_URL}/api/training/progress?query_id={query_id}", timeout=10)
        status = resp.json()
        print(f"Training status: {status}")
        
        if status.get('status') == 'failed':
            raise Exception(f"Training failed: {status.get('error', 'Unknown error')}")
        
        return status

    def check_accuracy(**context):
        model_id = context['dag_run'].conf.get('model_id')
        resp = requests.get(f"{GO_API_URL}/api/models/finetuned", timeout=10)
        models = resp.json().get('models', [])
        
        model = next((m for m in models if m['id'] == model_id), None)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        accuracy = model.get('accuracy', 0)
        print(f"Model {model_id} accuracy: {accuracy}")
        
        threshold = float(os.getenv('MIN_ACCURACY_THRESHOLD', '0.7'))
        if accuracy < threshold:
            print(f"Low accuracy {accuracy} < {threshold}, triggering retraining")
            context['task_instance'].xcom_push(key='needs_retraining', value=True)
        else:
            print(f"Accuracy OK: {accuracy}")
            context['task_instance'].xcom_push(key='needs_retraining', value=False)
        
        return accuracy

    def trigger_retry(**context):
        model_id = context['dag_run'].conf.get('model_id')
        file_ids = context['dag_run'].conf.get('file_ids', [])
        user_id = context['dag_run'].conf.get('user_id', '')
        needs_retraining = context['task_instance'].xcom_pull(task_ids='check_accuracy', key='needs_retraining')
        
        if needs_retraining:
            print(f"[AIRFLOW] Triggering retraining for model {model_id}")
            resp = requests.post(f"{GO_API_URL}/api/train/multi", 
                json={"file_ids": file_ids, "model_name": f"retrain_{model_id[:8]}"},
                headers={"X-User-ID": user_id},
                timeout=30
            )
            print(f"[AIRFLOW] Retraining triggered: {resp.json()}")
        else:
            print(f"[AIRFLOW] No retraining needed for model {model_id}")

    def notify_kafka(**context):
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            model_id = context['dag_run'].conf.get('model_id')
            accuracy = context['task_instance'].xcom_pull(task_ids='check_accuracy')
            producer.send('training_events', {
                'event': 'training_completed',
                'model_id': model_id,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            })
            producer.flush()
            print(f"Kafka event sent: training_completed model={model_id}")
        except Exception as e:
            print(f"Kafka notify failed (non-critical): {e}")

    t1 = PythonOperator(task_id='check_training_status', python_callable=check_training_status)
    t2 = PythonOperator(task_id='check_accuracy', python_callable=check_accuracy)
    t3 = PythonOperator(task_id='notify_kafka', python_callable=notify_kafka)

    t4 = PythonOperator(task_id="trigger_retry", python_callable=trigger_retry)
    t1 >> t2 >> t4 >> t3

# DAG 2: Scheduled connection sync
with DAG(
    'connection_sync',
    default_args=default_args,
    description='Nightly connection sync and retraining',
    schedule_interval='0 2 * * *',  # Her gece 02:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['sync'],
) as dag2:

    def sync_connections(**context):
        resp = requests.get(f"{GO_API_URL}/api/scheduler/status", timeout=30)
        print(f"Scheduler status: {resp.json()}")
        
        # Trigger sync for all real-time models
        resp2 = requests.post(f"{GO_API_URL}/api/scheduler/sync", timeout=60)
        print(f"Sync triggered: {resp2.json()}")

    t_sync = PythonOperator(task_id='sync_connections', python_callable=sync_connections)
    t_sync
