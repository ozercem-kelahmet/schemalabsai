import argparse
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def main():
    parser = argparse.ArgumentParser(description="Decrypt a SchemaLabs dedicated bundle (no inference)")
    parser.add_argument("--encrypted", required=True)
    parser.add_argument("--key", required=True, help="Hex key or path to key file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key_hex = args.key
    if os.path.exists(key_hex):
        with open(key_hex) as f:
            key_hex = f.read().strip()
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes (64 hex chars)")

    with open(args.encrypted, "rb") as f:
        blob = f.read()
    nonce, ct = blob[:12], blob[12:]

    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(nonce, ct, None)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(pt)
    print(f"Decrypted {len(pt)} bytes -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
