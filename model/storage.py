"""
Storage abstraction layer for Flask ML server.
Reads same env vars as Go backend:
  STORAGE_BACKEND=local|gcs|s3
  GCS_BUCKET=schemalabs-prod-us-central1
"""
import os
import io
import tempfile
import logging

log = logging.getLogger(__name__)

STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'local')
GCS_BUCKET = os.getenv('GCS_BUCKET', 'schemalabs-prod-us-central1')

_gcs_client = None

def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
        log.info(f"[STORAGE] GCS client initialized, bucket={GCS_BUCKET}")
    return _gcs_client

def upload(key, data):
    """Upload bytes or file-like object to storage"""
    if STORAGE_BACKEND == 'gcs':
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(key)
        if isinstance(data, (bytes, bytearray)):
            blob.upload_from_string(data)
        elif hasattr(data, 'read'):
            blob.upload_from_file(data)
        else:
            blob.upload_from_filename(str(data))
        log.info(f"[STORAGE] GCS uploaded: gs://{GCS_BUCKET}/{key}")
        return f"gs://{GCS_BUCKET}/{key}"
    else:
        # Local
        os.makedirs(os.path.dirname(key) if '/' in key else '.', exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            with open(key, 'wb') as f:
                f.write(data)
        elif hasattr(data, 'read'):
            with open(key, 'wb') as f:
                while True:
                    chunk = data.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        log.info(f"[STORAGE] Local saved: {key}")
        return key

def download(key):
    """Download from storage, returns file path (downloads to temp if GCS)"""
    if STORAGE_BACKEND == 'gcs':
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(key)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: {key}")
        # Download to temp file
        suffix = os.path.splitext(key)[1] or '.pt'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        blob.download_to_file(tmp)
        tmp.close()
        log.info(f"[STORAGE] GCS downloaded: {key} -> {tmp.name}")
        return tmp.name
    else:
        if not os.path.exists(key):
            raise FileNotFoundError(f"Local file not found: {key}")
        return key

def exists(key):
    """Check if file exists in storage"""
    if STORAGE_BACKEND == 'gcs':
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        return bucket.blob(key).exists()
    else:
        return os.path.exists(key)

def delete(key):
    """Delete file from storage"""
    if STORAGE_BACKEND == 'gcs':
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(key)
        if blob.exists():
            blob.delete()
            log.info(f"[STORAGE] GCS deleted: {key}")
    else:
        if os.path.exists(key):
            os.remove(key)

def user_key(user_id, category, filename):
    """Create user-scoped storage key"""
    return f"users/{user_id}/{category}/{filename}"

def shared_key(category, filename):
    """Create shared storage key"""
    return f"shared/{category}/{filename}"

def checkpoint_path(model_path, user_id=None):
    """Resolve checkpoint path — try storage, fallback local"""
    # Try multiple paths
    candidates = []
    if user_id:
        candidates.append(user_key(user_id, "checkpoints", model_path))
        candidates.append(user_key(user_id, "checkpoints", model_path + ".pt"))
    candidates.append(shared_key("base-models", model_path))
    candidates.append(shared_key("base-models", model_path + ".pt"))
    # Local fallbacks
    candidates.append(f"../checkpoints/{model_path}")
    candidates.append(f"../checkpoints/{model_path}.pt")
    candidates.append(f"checkpoints/{model_path}")
    candidates.append(f"checkpoints/{model_path}.pt")

    for path in candidates:
        try:
            if exists(path):
                return download(path)
        except:
            continue

    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

def save_checkpoint(data, filename, user_id=None):
    """Save checkpoint to storage"""
    if user_id:
        key = user_key(user_id, "checkpoints", filename)
    else:
        key = shared_key("base-models", filename)

    # Save locally first (torch needs file path)
    local_path = f"../checkpoints/{filename}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    import torch
    torch.save(data, local_path)

    # Upload to cloud storage
    if STORAGE_BACKEND != 'local':
        upload(key, local_path)
        log.info(f"[STORAGE] Checkpoint saved: {key}")

    return local_path

print(f"[STORAGE] Python storage module loaded: backend={STORAGE_BACKEND}")
