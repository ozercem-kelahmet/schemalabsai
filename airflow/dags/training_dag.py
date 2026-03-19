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
        
        # Training tamamlanana kadar bekle (max 30 dakika)
        import time
        max_wait = 1800
        waited = 0
        while waited < max_wait:
            try:
                resp = requests.get(
                    f"{GO_API_URL}/api/train/progress?query_id={query_id}",
                    timeout=10
                )
                if resp.status_code == 200 and resp.text:
                    status = resp.json()
                    current_status = status.get("status", "unknown")
                    print(f"[AIRFLOW] Training status: {current_status} epoch={status.get('epoch',0)}/{status.get('epochs',0)}")
                    
                    if current_status == "completed":
                        print(f"[AIRFLOW] Training completed!")
                        return status
                    elif current_status == "failed":
                        raise Exception(f"Training failed: {status.get('error', 'Unknown')}")
                    elif current_status == "idle" and waited > 60:
                        print(f"[AIRFLOW] Training idle after {waited}s, assuming completed")
                        return status
            except Exception as e:
                print(f"[AIRFLOW] Status check error: {e}")
            
            time.sleep(30)
            waited += 30
        
        raise Exception(f"Training timeout after {max_wait}s")

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
            try:
                resp = requests.post(
                    f"{GO_API_URL}/api/train/multi",
                    json={
                        "file_ids": file_ids,
                        "model_name": f"retrain_{model_id[:8]}",
                        "epochs": 0,
                        "batch_size": 0
                    },
                    headers={"X-User-ID": user_id, "Content-Type": "application/json"},
                    timeout=30
                )
                if resp.status_code == 200:
                    print(f"[AIRFLOW] Retraining triggered: {resp.json()}")
                else:
                    print(f"[AIRFLOW] Retraining failed: {resp.status_code} {resp.text}")
            except Exception as e:
                print(f"[AIRFLOW] Retraining error: {e}")
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
        # Scheduler status
        try:
            resp = requests.get(f"{GO_API_URL}/api/scheduler/status", timeout=30)
            if resp.status_code == 200 and resp.text:
                data = resp.json()
                print(f"Scheduler status: {data}")
            else:
                print(f"Scheduler status: {resp.status_code}")
        except Exception as e:
            print(f"Scheduler status check failed (non-critical): {e}")
        
        # Trigger sync for all real-time models
        try:
            resp2 = requests.post(f"{GO_API_URL}/api/scheduler/sync", timeout=60)
            if resp2.status_code == 200 and resp2.text:
                result = resp2.json()
                print(f"Sync triggered: {result}")
                # Sync bittikten sonra training_monitor DAG'ını tetikle
                synced_models = result.get("synced_models", [])
                for model in synced_models:
                    try:
                        airflow_resp = requests.post(
                            "http://localhost:8080/api/v1/dags/training_monitor/dagRuns",
                            json={"conf": {
                                "model_id": model.get("id", ""),
                                "query_id": model.get("id", ""),
                                "user_id": model.get("user_id", ""),
                                "file_ids": model.get("file_ids", [])
                            }},
                            auth=(os.getenv("AIRFLOW_ADMIN_USER", "admin"), os.getenv("AIRFLOW_ADMIN_PASSWORD", "")),
                            timeout=10
                        )
                        print(f"[AIRFLOW] training_monitor triggered for {model.get('id')}: {airflow_resp.status_code}")
                    except Exception as te:
                        print(f"[AIRFLOW] Trigger failed: {te}")
            else:
                print(f"Sync triggered: {resp2.status_code}")
        except Exception as e:
            print(f"Sync failed (non-critical): {e}")

    t_sync = PythonOperator(task_id='sync_connections', python_callable=sync_connections)
    t_sync

# DAG 3: Incremental learning pipeline
with DAG(
    'incremental_learning',
    default_args=default_args,
    description='Spark merge new data + retraining',
    schedule_interval=None,  # Event triggered
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['training', 'incremental'],
) as dag3:

    def merge_and_retrain(**context):
        model_id = context['dag_run'].conf.get('model_id')
        user_id = context['dag_run'].conf.get('user_id', '')
        new_file_ids = context['dag_run'].conf.get('new_file_ids', [])
        existing_file_ids = context['dag_run'].conf.get('existing_file_ids', [])
        
        all_file_ids = existing_file_ids + new_file_ids
        print(f"[INCREMENTAL] Model: {model_id}, files: {len(all_file_ids)} total ({len(new_file_ids)} new)")
        
        if not all_file_ids:
            print("[INCREMENTAL] No files, skipping")
            return
        
        # Go API üzerinden retraining tetikle
        try:
            resp = requests.post(
                f"{GO_API_URL}/api/train/multi",
                json={
                    "file_ids": all_file_ids,
                    "model_name": f"incremental_{model_id[:8]}",
                    "epochs": 0,
                    "batch_size": 0
                },
                headers={"X-User-ID": user_id, "Content-Type": "application/json"},
                timeout=30
            )
            if resp.status_code == 200:
                result = resp.json()
                print(f"[INCREMENTAL] Retraining triggered: {result.get('query_id')}")
                context['task_instance'].xcom_push(key='query_id', value=result.get('query_id'))
            else:
                raise Exception(f"Retraining failed: {resp.status_code} {resp.text}")
        except Exception as e:
            raise Exception(f"Incremental learning failed: {e}")

    def wait_and_verify(**context):
        import time
        query_id = context['task_instance'].xcom_pull(task_ids='merge_and_retrain', key='query_id')
        model_id = context['dag_run'].conf.get('model_id')
        
        if not query_id:
            print("[INCREMENTAL] No query_id, skipping")
            return
        
        # Training tamamlanana kadar bekle
        max_wait = 3600
        waited = 0
        while waited < max_wait:
            try:
                resp = requests.get(f"{GO_API_URL}/api/train/progress?query_id={query_id}", timeout=10)
                if resp.status_code == 200 and resp.text:
                    status = resp.json()
                    current = status.get("status", "unknown")
                    print(f"[INCREMENTAL] Status: {current} epoch={status.get('epoch',0)}/{status.get('epochs',0)}")
                    if current == "completed":
                        accuracy = status.get("accuracy", 0)
                        print(f"[INCREMENTAL] Done! accuracy={accuracy:.1f}%")
                        return
                    elif current == "failed":
                        raise Exception(f"Training failed: {status.get('error')}")
            except Exception as e:
                print(f"[INCREMENTAL] Check error: {e}")
            time.sleep(30)
            waited += 30
        
        raise Exception(f"Incremental learning timeout")

    t_merge = PythonOperator(task_id='merge_and_retrain', python_callable=merge_and_retrain)
    t_verify = PythonOperator(task_id='wait_and_verify', python_callable=wait_and_verify)
    t_merge >> t_verify
