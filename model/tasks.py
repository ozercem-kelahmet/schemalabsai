from celery import Celery
import os
import sys
from dotenv import load_dotenv

load_dotenv('../.env')

# Redis config
redis_url = os.getenv('REDIS_URL')
redis_password = os.getenv('REDIS_PASSWORD')
host, port = redis_url.split(':')
broker_url = f'redis://:{redis_password}@{host}:{port}/0'
backend_url = f'redis://:{redis_password}@{host}:{port}/1'

# Celery app
app = Celery('schemalabs_tasks',
             broker=broker_url,
             backend=backend_url)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
)

@app.task(bind=True)
def train_model_task(self, user_file_path, target_column, model_name, epochs=20):
    """Asenkron training task"""
    import time
    from train import train_model  # Mevcut train fonksiyonun
    
    try:
        # Progress callback
        def progress_callback(epoch, total_epochs, acc, loss):
            progress = int((epoch / total_epochs) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': epoch,
                    'total': total_epochs,
                    'accuracy': acc,
                    'loss': loss,
                    'progress': progress
                }
            )
        
        # Training başlat
        result = train_model(
            user_file_path=user_file_path,
            target_column=target_column,
            model_name=model_name,
            epochs=epochs,
            progress_callback=progress_callback
        )
        
        return {
            'status': 'completed',
            'result': result
        }
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise
