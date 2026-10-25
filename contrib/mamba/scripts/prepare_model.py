from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "state-spaces/mamba-130m-hf"
REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
SAFETENSORS_SHA256 = "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the pinned official Mamba-130m checkpoint")
    parser.add_argument("--output-dir", default="build/mamba-model")
    args = parser.parse_args()
    destination = Path(args.output_dir)
    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=("*.json", "model.safetensors"),
    )
    checkpoint = destination / "model.safetensors"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != SAFETENSORS_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {SAFETENSORS_SHA256}, got {digest}")
    print(f"verified {MODEL_ID}@{REVISION}: {digest}")


if __name__ == "__main__":
    main()
