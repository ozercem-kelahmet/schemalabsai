from flask import jsonify
from tasks import train_model_task

def finetune_async(request):
    """Asenkron fine-tune endpoint"""
    try:
        data = request.get_json()
        user_file_path = data.get('file_path')
        target_column = data.get('target_column')
        model_name = data.get('model_name', 'user_model')
        epochs = int(data.get('epochs', 20))
        
        # Celery task'ı başlat (hemen döner)
        task = train_model_task.delay(
            user_file_path=user_file_path,
            target_column=target_column,
            model_name=model_name,
            epochs=epochs
        )
        
        return jsonify({
            'status': 'started',
            'task_id': task.id,
            'message': 'Training started in background'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_training_progress(task_id):
    """Training progress endpoint"""
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=train_model_task.app)
    
    if result.state == 'PENDING':
        response = {
            'state': result.state,
            'progress': 0,
            'status': 'Waiting...'
        }
    elif result.state == 'PROGRESS':
        response = {
            'state': result.state,
            'progress': result.info.get('progress', 0),
            'current': result.info.get('current', 0),
            'total': result.info.get('total', 0),
            'accuracy': result.info.get('accuracy', 0),
            'loss': result.info.get('loss', 0),
            'status': 'Training...'
        }
    elif result.state == 'SUCCESS':
        response = {
            'state': result.state,
            'progress': 100,
            'result': result.result,
            'status': 'Completed!'
        }
    else:
        # FAILURE or other
        response = {
            'state': result.state,
            'progress': 0,
            'status': str(result.info)
        }
    
    return jsonify(response)
