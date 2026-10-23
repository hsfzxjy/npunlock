from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CONTRIB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRIB_ROOT))

from xgboost_npu.generate import generate_kernel_source
from xgboost_npu.model import load_ensemble
from xgboost_npu.packing import batch_capacity


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ACT-SHAVE C kernel from XGBoost JSON")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--transport", choices=("f16", "f32"), default="f16")
    return parser.parse_args()


def main() -> int:
    args = options()
    ensemble = load_ensemble(args.model)
    source = generate_kernel_source(ensemble, transport=args.transport)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(source)
    manifest = {
        "schema": "npunlock-xgboost-kernel-v1",
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(source).hexdigest(),
        "feature_count": ensemble.feature_count,
        "tree_count": len(ensemble.trees),
        "node_count": ensemble.node_count,
        "max_depth": ensemble.max_depth,
        "maximum_batch_size": batch_capacity(ensemble.feature_count, transport=args.transport),
        "transport": args.transport,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
