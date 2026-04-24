SchemaLabs Dedicated Deployment Runtime

This package runs encrypted SchemaLabs model checkpoints on your own GPU infrastructure.

Requirements:

- NVIDIA GPU with >=16 GB VRAM (T4, V100, A100, H100 supported)
- CUDA 12.1+ drivers
- Python 3.10+
- Docker with nvidia-container-toolkit (optional, for containerized deployment)

Files:

- model_name_vN.enc : Encrypted checkpoint (AES-256-GCM)
- key.txt : 64-char hex decryption key (delivered out-of-band)
- inference.py : Decryption + model load runtime
- decrypt_only.py : Stand-alone decryption utility
- Dockerfile and requirements.txt : Container recipe
- README.md : This file

Quick start (bare-metal):

    pip install -r requirements.txt
    python inference.py --encrypted my_model_v1.enc --key key.txt --device cuda

Docker:

    docker build -t schemalabs-dedicated .
    docker run --gpus all --rm \
      -v PWD/my_model_v1.enc:/bundle/model.enc:ro \
      -v PWD/key.txt:/bundle/key.txt:ro \
      schemalabs-dedicated \
      --encrypted /bundle/model.enc --key /bundle/key.txt

Replace PWD with your actual working directory path.

Air-gapped deployment:

1. Download bundle plus runtime package on an internet-connected workstation
2. Transfer via approved media (USB, one-way diode) to the air-gapped environment
3. Pre-build the Docker image with docker save and docker load, or install dependencies offline
4. Run inference without network egress

The encrypted bundle never contacts SchemaLabs infrastructure at runtime. No telemetry, no call-home.

Security:

- The decryption key MUST be delivered separately from the encrypted bundle
- Never commit key.txt to any repository
- Rotate keys periodically via the SchemaLabs dashboard
- After decryption, the runtime deletes the plaintext checkpoint from disk unless --keep-decrypted is passed
- Encrypted bundle integrity is verifiable via the X-Bundle-SHA256 response header on download

Compliance:

Each download is audit-logged on the SchemaLabs side (timestamp, IP, user agent). MSA with IP protection clauses applies to all deployed artifacts. Checkpoint version pinning is enforced -- rotating keys produces a new versioned bundle.

Support:

Enterprise support: dedicated CSM. Contact your account manager.
