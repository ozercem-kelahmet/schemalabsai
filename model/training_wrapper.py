import json
import traceback
from async_training import redis_client

def wrap_finetune_for_async(finetune_func, task_id, request_data):
    """Mevcut finetune fonksiyonunu async wrapper ile çalıştır"""
    
    try:
        # Training başladı
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'STARTED',
            'progress': 0,
            'status': 'Initializing...'
        }))
        
        # Mevcut finetune fonksiyonunu çağır
        # request_data içinde form data var
        result = finetune_func(request_data, task_id)
        
        # Başarıyla tamamlandı
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'SUCCESS',
            'progress': 100,
            'result': result,
            'status': 'Completed!'
        }))
        
    except Exception as e:
        # Hata
        redis_client.setex(f'training:{task_id}', 3600, json.dumps({
            'state': 'FAILURE',
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
