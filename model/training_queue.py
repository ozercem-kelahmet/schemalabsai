"""
Training Queue System
- Sistem kaynaklarini olcer (CPU, RAM, GPU)
- Dinamik max concurrent training hesaplar
- Kuyruk yonetimi yapar
- Slot acilinca otomatik baslatir
"""
import threading
import time
import json
import os
import psutil
from collections import deque
from datetime import datetime

# GPU memory check
def get_gpu_info():
    """GPU VRAM bilgisi doner (MB cinsinden)"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return {
                'total_mb': int(parts[0].strip()),
                'used_mb': int(parts[1].strip()),
                'free_mb': int(parts[2].strip()),
                'available': True
            }
    except:
        pass
    return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'available': False}

def get_system_resources():
    """Sistem kaynaklarini olc"""
    cpu_count = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    gpu = get_gpu_info()
    return {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'cpu_available': max(0, 100 - cpu_percent),
        'ram_total_gb': round(mem.total / (1024**3), 1),
        'ram_available_gb': round(mem.available / (1024**3), 1),
        'ram_percent_used': mem.percent,
        'gpu': gpu
    }

def calculate_max_concurrent(resources):
    """Sistem kaynaklarina gore max concurrent training hesapla"""
    # Her training icin gereken minimum kaynaklar
    PER_TRAINING_RAM_GB = 1.5   # ~1.5GB RAM per training
    PER_TRAINING_CPU = 1.5      # ~1.5 CPU core per training
    PER_TRAINING_GPU_MB = 2000  # ~2GB VRAM per training (T4 icin)

    # RAM bazli limit
    ram_limit = int(resources['ram_available_gb'] / PER_TRAINING_RAM_GB)

    # CPU bazli limit
    cpu_available_cores = resources['cpu_count'] * (resources['cpu_available'] / 100)
    cpu_limit = int(cpu_available_cores / PER_TRAINING_CPU)

    # GPU bazli limit
    if resources['gpu']['available'] and resources['gpu']['free_mb'] > 0:
        gpu_limit = int(resources['gpu']['free_mb'] / PER_TRAINING_GPU_MB)
        gpu_limit = max(1, gpu_limit)
    else:
        gpu_limit = 999  # GPU yoksa limitleme

    # En dusuk limiti al, minimum 1
    max_concurrent = max(1, min(ram_limit, cpu_limit, gpu_limit))

    # Hard cap - asiri yuk koruması

    max_concurrent = min(max_concurrent, 10)

    return max_concurrent

class TrainingQueue:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._active_trainings = {}   # task_id -> info
        self._queue = deque()          # queued items
        self._queue_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self._worker_thread.start()
        print("[TrainingQueue] Initialized")

    def _queue_worker(self):
        """Arka planda calisir, slot acilinca kuyruktan baslatir"""
        while True:
            try:
                time.sleep(3)
                with self._queue_lock:
                    if len(self._queue) == 0:
                        continue
                    # Bitenleri temizle
                    self._cleanup_finished()
                    # Kapasite kontrol
                    resources = get_system_resources()
                    max_c = calculate_max_concurrent(resources)
                    active = len(self._active_trainings)
                    available_slots = max_c - active
                    if available_slots > 0:
                        # Siradan al ve baslat
                        for _ in range(min(available_slots, len(self._queue))):
                            item = self._queue.popleft()
                            self._start_training(item)
            except Exception as e:
                print(f"[TrainingQueue] Worker error: {e}")

    def _cleanup_finished(self):
        """Bitmis training'leri temizle"""
        finished = []
        for task_id, info in self._active_trainings.items():
            if info.get('thread') and not info['thread'].is_alive():
                finished.append(task_id)
        for task_id in finished:
            del self._active_trainings[task_id]

    def _start_training(self, item):
        """Queue'daki item'i baslat"""
        task_id = item['task_id']
        train_func = item['func']
        args = item.get('args', ())
        kwargs = item.get('kwargs', {})

        # Session guncelle
        from server import get_session, save_session
        session = get_session(task_id)
        session['status'] = 'starting'
        session['queued'] = False
        save_session(task_id, session)

        def wrapper():
            try:
                train_func(*args, **kwargs)
            except Exception as e:
                print(f"[TrainingQueue] Training {task_id} failed: {e}")

        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        self._active_trainings[task_id] = {
            'thread': t,
            'started_at': datetime.now().isoformat(),
            'user_id': item.get('user_id', ''),
            'task_id': task_id
        }
        print(f"[TrainingQueue] Started {task_id} from queue (active={len(self._active_trainings)})")

    def submit(self, task_id, train_func, user_id='', args=(), kwargs={}):
        """Training submit et. Kapasite varsa hemen basla, yoksa kuyruge al."""
        with self._queue_lock:
            self._cleanup_finished()
            resources = get_system_resources()
            max_c = calculate_max_concurrent(resources)
            active = len(self._active_trainings)

            if active < max_c:
                # Hemen baslat
                item = {'task_id': task_id, 'func': train_func, 'user_id': user_id, 'args': args, 'kwargs': kwargs}
                self._start_training(item)
                return {
                    'status': 'started',
                    'task_id': task_id,
                    'position': 0,
                    'active': active + 1,
                    'max_concurrent': max_c,
                    'message': 'Training started'
                }
            else:
                # Kuyruge al
                position = len(self._queue) + 1
                self._queue.append({
                    'task_id': task_id,
                    'func': train_func,
                    'user_id': user_id,
                    'args': args,
                    'kwargs': kwargs,
                    'queued_at': datetime.now().isoformat()
                })

                # Session'a queued durumu yaz
                from server import get_session, save_session
                session = get_session(task_id)
                session['status'] = 'queued'
                session['queued'] = True
                session['queue_position'] = position
                save_session(task_id, session)

                return {
                    'status': 'queued',
                    'task_id': task_id,
                    'position': position,
                    'active': active,
                    'max_concurrent': max_c,
                    'message': f'Server busy. Training queued at position {position}. Will start automatically.'
                }

    def get_status(self):
        """Queue durumu"""
        with self._queue_lock:
            self._cleanup_finished()
            resources = get_system_resources()
            max_c = calculate_max_concurrent(resources)
            return {
                'active_trainings': len(self._active_trainings),
                'queued': len(self._queue),
                'max_concurrent': max_c,
                'resources': resources,
                'active_ids': list(self._active_trainings.keys()),
                'queued_ids': [item['task_id'] for item in self._queue]
            }

# Singleton
training_queue = TrainingQueue()
