import threading
import time
import redis
import json
import os
from dotenv import load_dotenv

load_dotenv('../.env')

# Redis connection
redis_url = os.getenv('REDIS_URL')
redis_password = os.getenv('REDIS_PASSWORD')
host, port = redis_url.split(':')

redis_client = redis.Redis(
    host=host,
    port=int(port),
    password=redis_password,
    decode_responses=True
)

def run_training_async(training_func, task_id, *args, **kwargs):
    """Thread'de training çalıştır, progress'i Redis'e kaydet"""
    
    def progress_callback(epoch, total_epochs, acc, loss):
        """Training progress'i Redis'e kaydet"""
        progress_data = {
            'state': 'PROGRESS',
            'current': epoch,
            'total': total_epochs,
            'accuracy': round(acc, 2),
            'loss': round(loss, 4),
            'progress': int((epoch / total_epochs) * 100)
        }
        redis_client.setex(f'training:{task_id}', 3600, json.dumps(progress_data))
    
    try:
        # Training başladı
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'STARTED',
            'progress': 0
        }))
        
        # Training fonksiyonunu çalıştır (callback ile)
        result = training_func(progress_callback=progress_callback, *args, **kwargs)
        
        # Başarıyla tamamlandı
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'SUCCESS',
            'progress': 100,
            'result': result
        }))
        
    except Exception as e:
        # Hata oluştu
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'FAILURE',
            'error': str(e)
        }))

def start_training_thread(training_func, task_id, *args, **kwargs):
    """Training'i arka planda thread'de başlat"""
    thread = threading.Thread(
        target=run_training_async,
        args=(training_func, task_id) + args,
        kwargs=kwargs,
        daemon=True
    )
    thread.start()
    return task_id

def get_training_status(task_id):
    """Redis'ten training status'ü al"""
    data = redis_client.get(f'training:{task_id}')
    if data:
        return json.loads(data)
    return {'state': 'PENDING', 'progress': 0}
