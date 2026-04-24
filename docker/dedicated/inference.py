import argparse
import json
import os
import sys
from pathlib import Path

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def decrypt_checkpoint(encrypted_path: str, key_hex: str, output_path: str) -> None:
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError("Decryption key must be 32 bytes (64 hex chars)")

    with open(encrypted_path, "rb") as f:
        blob = f.read()

    nonce, ciphertext = blob[:12], blob[12:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(plaintext)


def load_model(checkpoint_path, device="cuda"):
    if not torch.cuda.is_available() and device == "cuda":
        print("[WARN] CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return state, device


def run_inference(state, inputs, device):
    if isinstance(state, dict) and "state_dict" in state:
        print(f"[INFO] Loaded state_dict with {len(state['state_dict'])} tensors", file=sys.stderr)
    else:
        print(f"[INFO] Loaded checkpoint type: {type(state).__name__}", file=sys.stderr)
    print("[INFO] Model loaded. Wire this to your inference pipeline.", file=sys.stderr)
    return {"status": "loaded", "device": device, "inputs_received": len(inputs) if inputs else 0}


def main():
    parser = argparse.ArgumentParser(description="SchemaLabs dedicated deployment inference runtime")
    parser.add_argument("--encrypted", required=True)
    parser.add_argument("--key", required=True, help="Hex key or path to key file")
    parser.add_argument("--output", default="/tmp/decrypted_checkpoint.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input")
    parser.add_argument("--keep-decrypted", action="store_true")
    args = parser.parse_args()

    key_hex = args.key
    if os.path.exists(key_hex):
        with open(key_hex) as f:
            key_hex = f.read().strip()

    print(f"[INFO] Decrypting {args.encrypted} -> {args.output}", file=sys.stderr)
    decrypt_checkpoint(args.encrypted, key_hex, args.output)

    state, device = load_model(args.output, args.device)

    inputs = None
    if args.input:
        with open(args.input) as f:
            inputs = json.load(f)

    result = run_inference(state, inputs, device)

    if not args.keep_decrypted:
        try:
            os.remove(args.output)
            print(f"[INFO] Wiped decrypted checkpoint: {args.output}", file=sys.stderr)
        except OSError:
            pass

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
