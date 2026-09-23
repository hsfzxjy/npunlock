"""Create and verify a reproducible npunlock porting probe bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "npunlock.porting.probe.v1"
VERIFY_SCHEMA = "npunlock.porting.probe.verify.v1"
GRAPH_FILE = "graph.blob"
INPUT_FILE = "input-0.bin"
EXPECTED_FILE = "expected-output-0.bin"
MANIFEST_FILE = "manifest.json"
INPUT_VALUES = (
    -4.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fp16_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def _file_record(name: str, data: bytes) -> dict[str, object]:
    return {"file": name, "size": len(data), "sha256": _sha256(data)}


def create_bundle(
    graph_path: Path,
    output_dir: Path,
    *,
    producer_platform: str,
    device: str,
    driver_version: str,
    graph_compiler_version: str,
) -> Path:
    graph = graph_path.read_bytes()
    if not graph:
        raise ValueError("graph blob must not be empty")
    output_dir.mkdir(parents=True, exist_ok=False)

    input_data = _fp16_bytes(INPUT_VALUES)
    expected_data = _fp16_bytes(tuple(value + 1.0 for value in INPUT_VALUES))
    graph_record = _file_record(GRAPH_FILE, graph)
    input_record = {
        **_file_record(INPUT_FILE, input_data),
        "argument_index": 0,
        "dtype": "f16",
        "shape": [1, len(INPUT_VALUES)],
    }
    output_record = {
        **_file_record(EXPECTED_FILE, expected_data),
        "argument_index": 1,
        "dtype": "f16",
        "shape": [1, len(INPUT_VALUES)],
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "purpose": "Test loading and exact FP16 add-one execution of one prepatched native graph.",
        "producer": {
            "platform": producer_platform,
            "device": device,
            "driver_version": driver_version,
            "graph_compiler_version": graph_compiler_version,
        },
        "graph": graph_record,
        "inputs": [input_record],
        "expected_outputs": [output_record],
        "oracle": {
            "name": "fp16_add1_exact",
            "input_argument_index": 0,
            "output_argument_index": 1,
        },
    }

    (output_dir / GRAPH_FILE).write_bytes(graph)
    (output_dir / INPUT_FILE).write_bytes(input_data)
    (output_dir / EXPECTED_FILE).write_bytes(expected_data)
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _single_record(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must contain exactly one tensor")
    return _object(value[0], f"{name}[0]")


def _bundle_file(bundle_dir: Path, record: Mapping[str, Any], expected_name: str) -> bytes:
    name = record.get("file")
    if (
        name != expected_name
        or PurePosixPath(expected_name).is_absolute()
        or ".." in PurePosixPath(expected_name).parts
    ):
        raise ValueError(f"unexpected bundle file name: {name!r}")
    data = (bundle_dir / expected_name).read_bytes()
    if record.get("size") != len(data) or record.get("sha256") != _sha256(data):
        raise ValueError(f"size or SHA-256 mismatch for {expected_name}")
    return data


def verify_bundle(bundle_dir: Path) -> dict[str, object]:
    manifest = _object(json.loads((bundle_dir / MANIFEST_FILE).read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported probe-bundle schema")
    producer = _object(manifest.get("producer"), "producer")
    required_provenance = ("platform", "device", "driver_version", "graph_compiler_version")
    if any(not isinstance(producer.get(key), str) or not producer[key] for key in required_provenance):
        raise ValueError("producer provenance fields must be nonempty strings")

    graph_record = _object(manifest.get("graph"), "graph")
    input_record = _single_record(manifest.get("inputs"), "inputs")
    output_record = _single_record(manifest.get("expected_outputs"), "expected_outputs")
    graph = _bundle_file(bundle_dir, graph_record, GRAPH_FILE)
    input_data = _bundle_file(bundle_dir, input_record, INPUT_FILE)
    expected_data = _bundle_file(bundle_dir, output_record, EXPECTED_FILE)

    if (
        input_record.get("argument_index") != 0
        or output_record.get("argument_index") != 1
        or input_record.get("dtype") != "f16"
        or output_record.get("dtype") != "f16"
        or input_record.get("shape") != [1, len(INPUT_VALUES)]
        or output_record.get("shape") != [1, len(INPUT_VALUES)]
    ):
        raise ValueError("probe tensor contract is not the fixed 1x16 FP16 contract")
    oracle = _object(manifest.get("oracle"), "oracle")
    if oracle != {
        "name": "fp16_add1_exact",
        "input_argument_index": 0,
        "output_argument_index": 1,
    }:
        raise ValueError("unsupported probe oracle")
    actual_inputs = struct.unpack(f"<{len(INPUT_VALUES)}e", input_data)
    oracle_output = _fp16_bytes(tuple(value + 1.0 for value in actual_inputs))
    if expected_data != oracle_output:
        raise ValueError("expected output does not match the exact FP16 add-one oracle")

    expected_files = {MANIFEST_FILE, GRAPH_FILE, INPUT_FILE, EXPECTED_FILE}
    actual_entries = {path.name for path in bundle_dir.iterdir()}
    if actual_entries != expected_files:
        raise ValueError("probe bundle contains missing or unexpected files")
    return {
        "schema": VERIFY_SCHEMA,
        "bundle": str(bundle_dir),
        "graph_sha256": _sha256(graph),
        "element_count": len(INPUT_VALUES),
        "oracle": "fp16_add1_exact",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a deterministic FP16 add-one bundle")
    create.add_argument("--graph", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--producer-platform", required=True)
    create.add_argument("--device", required=True)
    create.add_argument("--driver-version", required=True)
    create.add_argument("--graph-compiler-version", required=True)
    verify = subparsers.add_parser("verify", help="verify hashes, tensor contract, and host oracle")
    verify.add_argument("bundle", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        manifest = create_bundle(
            args.graph,
            args.output_dir,
            producer_platform=args.producer_platform,
            device=args.device,
            driver_version=args.driver_version,
            graph_compiler_version=args.graph_compiler_version,
        )
        print(manifest)
    else:
        print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
