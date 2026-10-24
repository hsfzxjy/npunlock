from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .model import Ensemble, load_ensemble
from .packing import HEADER_WORDS, OUTPUT_WORDS_PER_SAMPLE, Predictions


RUNTIME_CHUNK_ELEMENTS = 1024
RUNTIME_LOGICAL_ELEMENTS = 4096
RUNTIME_REPLICAS = 4
RUNTIME_MODEL_HEADER_ELEMENTS = 5
RUNTIME_MODEL_MAGIC = 372020
RUNTIME_MAX_FEATURES = 63
RUNTIME_MAX_NODES_PER_TREE = 16
RUNTIME_MAX_TOTAL_NODES = (RUNTIME_CHUNK_ELEMENTS - RUNTIME_MODEL_HEADER_ELEMENTS) // 2

_RUNTIME_CACHE_SCHEMA = "npunlock-xgboost-runtime-cache-v1"
_MOVITOOLS_DEPENDENCIES = (
    "bin/moviCompile64.dll",
    "bin/moviAsm64.dll",
    "bin/moviLLD64.dll",
    "lib/mlibm.a",
)
_NATIVE_RUNTIME_DEPENDENCIES = (
    "npunlock.dll",
    "npunlock_worker.exe",
)

_FEATURE_BITS = 6
_CHILD_BITS = 4
_FEATURE_MASK = (1 << _FEATURE_BITS) - 1
_CHILD_MASK = (1 << _CHILD_BITS) - 1
_LEAF_FEATURE = _FEATURE_MASK
_LEFT_SHIFT = _FEATURE_BITS
_RIGHT_SHIFT = _LEFT_SHIFT + _CHILD_BITS
_DEFAULT_SHIFT = _RIGHT_SHIFT + _CHILD_BITS
_TREE_SIZE_SHIFT = _DEFAULT_SHIFT + 1


RUNTIME_KERNEL_SOURCE: bytes = (
    f"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define XGB_CHUNK_ELEMENTS {RUNTIME_CHUNK_ELEMENTS}u
#define XGB_FEATURE_HEADER {HEADER_WORDS}u
#define XGB_OUTPUT_WORDS {OUTPUT_WORDS_PER_SAMPLE}u
#define XGB_MODEL_HEADER {RUNTIME_MODEL_HEADER_ELEMENTS}u
#define XGB_MODEL_MAGIC {RUNTIME_MODEL_MAGIC}u
#define XGB_MAX_FEATURES {RUNTIME_MAX_FEATURES}u
#define XGB_MAX_TREE_NODES {RUNTIME_MAX_NODES_PER_TREE}u
#define XGB_MAX_TOTAL_NODES {RUNTIME_MAX_TOTAL_NODES}u
#define XGB_FEATURE_MASK {_FEATURE_MASK}u
#define XGB_CHILD_MASK {_CHILD_MASK}u
#define XGB_LEAF_FEATURE {_LEAF_FEATURE}u
#define XGB_LEFT_SHIFT {_LEFT_SHIFT}u
#define XGB_RIGHT_SHIFT {_RIGHT_SHIFT}u
#define XGB_DEFAULT_SHIFT {_DEFAULT_SHIFT}u
#define XGB_TREE_SIZE_SHIFT {_TREE_SIZE_SHIFT}u

float expf(float value);

static int xgb_exact_uint(float value, unsigned maximum, unsigned *result) {{
    unsigned converted;
    if (!(value >= 0.0f) || value > (float)maximum) {{
        return 0;
    }}
    converted = (unsigned)value;
    if ((float)converted != value) {{
        return 0;
    }}
    *result = converted;
    return 1;
}}

static float xgb_sigmoid(float value) {{
    if (value >= 0.0f) {{
        return 1.0f / (1.0f + expf(-value));
    }}
    float exponential = expf(value);
    return exponential / (1.0f + exponential);
}}

static int xgb_margin(
    const float *features,
    const float *model,
    unsigned sample,
    unsigned feature_count,
    unsigned tree_count,
    unsigned node_count,
    float *result) {{
    unsigned root = 0u;
    float margin = model[4];
    for (unsigned tree = 0u; tree < tree_count; ++tree) {{
        unsigned root_metadata;
        unsigned tree_nodes;
        unsigned node = 0u;
        int found_leaf = 0;
        if (root >= node_count || !xgb_exact_uint(
                model[XGB_MODEL_HEADER + root * 2u], 0x7ffffu, &root_metadata)) {{
            return 0;
        }}
        tree_nodes = ((root_metadata >> XGB_TREE_SIZE_SHIFT) & XGB_CHILD_MASK) + 1u;
        if (root + tree_nodes > node_count) {{
            return 0;
        }}
        for (unsigned step = 0u; step < XGB_MAX_TREE_NODES; ++step) {{
            unsigned metadata;
            unsigned feature;
            unsigned word;
            float split_or_leaf;
            if (node >= tree_nodes) {{
                return 0;
            }}
            word = XGB_MODEL_HEADER + (root + node) * 2u;
            if (!xgb_exact_uint(model[word], 0x7ffffu, &metadata)) {{
                return 0;
            }}
            feature = metadata & XGB_FEATURE_MASK;
            split_or_leaf = model[word + 1u];
            if (feature == XGB_LEAF_FEATURE) {{
                margin += split_or_leaf;
                found_leaf = 1;
                break;
            }}
            if (feature >= feature_count) {{
                return 0;
            }}
            unsigned left = (metadata >> XGB_LEFT_SHIFT) & XGB_CHILD_MASK;
            unsigned right = (metadata >> XGB_RIGHT_SHIFT) & XGB_CHILD_MASK;
            float value = features[
                XGB_FEATURE_HEADER + sample * feature_count + feature
            ];
            if (left >= tree_nodes || right >= tree_nodes) {{
                return 0;
            }}
            if (value != value) {{
                node = ((metadata >> XGB_DEFAULT_SHIFT) & 1u) != 0u ? left : right;
            }} else {{
                node = value < split_or_leaf ? left : right;
            }}
        }}
        if (!found_leaf) {{
            return 0;
        }}
        root += tree_nodes;
    }}
    if (root != node_count) {{
        return 0;
    }}
    *result = margin;
    return 1;
}}

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    const float *features;
    const float *model;
    float *output;
    unsigned sample_count;
    unsigned feature_count;
    unsigned magic;
    unsigned tree_count;
    unsigned node_count;
    unsigned model_feature_count;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count != XGB_CHUNK_ELEMENTS) {{
        return;
    }}
    features = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    model = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
    output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
    if (!xgb_exact_uint(features[0],
            (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / 1u, &sample_count) ||
        !xgb_exact_uint(features[1], XGB_MAX_FEATURES, &feature_count) ||
        feature_count == 0u ||
        sample_count > (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / feature_count ||
        sample_count > (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / XGB_OUTPUT_WORDS ||
        !xgb_exact_uint(model[0], XGB_MODEL_MAGIC, &magic) || magic != XGB_MODEL_MAGIC ||
        !xgb_exact_uint(model[1], 64u, &tree_count) || tree_count == 0u ||
        !xgb_exact_uint(model[2], XGB_MAX_TOTAL_NODES, &node_count) || node_count == 0u ||
        !xgb_exact_uint(model[3], XGB_MAX_FEATURES, &model_feature_count) ||
        model_feature_count != feature_count) {{
        return;
    }}
    for (unsigned index = 0u; index < XGB_CHUNK_ELEMENTS; ++index) {{
        output[index] = 0.0f;
    }}
    output[0] = (float)sample_count;
    output[1] = (float)feature_count;
    output[2] = (float)XGB_OUTPUT_WORDS;
    output[3] = (float)tree_count;
    for (unsigned sample = 0u; sample < sample_count; ++sample) {{
        float margin;
        float *record = output + XGB_FEATURE_HEADER + sample * XGB_OUTPUT_WORDS;
        if (!xgb_margin(
                features,
                model,
                sample,
                feature_count,
                tree_count,
                node_count,
                &margin)) {{
            output[3] = -1.0f;
            return;
        }}
        float probability = xgb_sigmoid(margin);
        record[0] = margin;
        record[1] = probability;
        record[2] = probability >= 0.5f ? 1.0f : 0.0f;
    }}
}}
""".encode(
        "utf-8"
    )
)


RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE: bytes = (
    f"""
#include <npunlock/npu3720_kernel.h>

#define XGB_CHUNK_ELEMENTS {RUNTIME_CHUNK_ELEMENTS}u
#define XGB_FEATURE_HEADER {HEADER_WORDS}u
#define XGB_MODEL_HEADER {RUNTIME_MODEL_HEADER_ELEMENTS}u
#define XGB_MODEL_MAGIC {RUNTIME_MODEL_MAGIC}u
#define XGB_MAX_FEATURES {RUNTIME_MAX_FEATURES}u
#define XGB_MAX_TREE_NODES {RUNTIME_MAX_NODES_PER_TREE}u
#define XGB_MAX_TOTAL_NODES {RUNTIME_MAX_TOTAL_NODES}u
#define XGB_FEATURE_MASK {_FEATURE_MASK}u
#define XGB_CHILD_MASK {_CHILD_MASK}u
#define XGB_LEAF_FEATURE {_LEAF_FEATURE}u
#define XGB_LEFT_SHIFT {_LEFT_SHIFT}u
#define XGB_RIGHT_SHIFT {_RIGHT_SHIFT}u
#define XGB_DEFAULT_SHIFT {_DEFAULT_SHIFT}u
#define XGB_TREE_SIZE_SHIFT {_TREE_SIZE_SHIFT}u

static int xgb_exact_uint(float value, unsigned maximum, unsigned *result) {{
    unsigned converted;
    if (!(value >= 0.0f) || value > (float)maximum) {{
        return 0;
    }}
    converted = (unsigned)value;
    if ((float)converted != value) {{
        return 0;
    }}
    *result = converted;
    return 1;
}}

static int xgb_margin(
    const float *features,
    const float *model,
    unsigned sample,
    unsigned feature_count,
    unsigned tree_count,
    unsigned node_count,
    float *result) {{
    unsigned root = 0u;
    float margin = model[4];
    for (unsigned tree = 0u; tree < tree_count; ++tree) {{
        unsigned root_metadata;
        unsigned tree_nodes;
        unsigned node = 0u;
        int found_leaf = 0;
        if (root >= node_count || !xgb_exact_uint(
                model[XGB_MODEL_HEADER + root * 2u], 0x7ffffu, &root_metadata)) {{
            return 0;
        }}
        tree_nodes = ((root_metadata >> XGB_TREE_SIZE_SHIFT) & XGB_CHILD_MASK) + 1u;
        if (root + tree_nodes > node_count) {{
            return 0;
        }}
        for (unsigned step = 0u; step < XGB_MAX_TREE_NODES; ++step) {{
            unsigned metadata;
            unsigned feature;
            unsigned word;
            float split_or_leaf;
            if (node >= tree_nodes) {{
                return 0;
            }}
            word = XGB_MODEL_HEADER + (root + node) * 2u;
            if (!xgb_exact_uint(model[word], 0x7ffffu, &metadata)) {{
                return 0;
            }}
            feature = metadata & XGB_FEATURE_MASK;
            split_or_leaf = model[word + 1u];
            if (feature == XGB_LEAF_FEATURE) {{
                margin += split_or_leaf;
                found_leaf = 1;
                break;
            }}
            if (feature >= feature_count) {{
                return 0;
            }}
            unsigned left = (metadata >> XGB_LEFT_SHIFT) & XGB_CHILD_MASK;
            unsigned right = (metadata >> XGB_RIGHT_SHIFT) & XGB_CHILD_MASK;
            float value = features[
                XGB_FEATURE_HEADER + sample * feature_count + feature
            ];
            if (left >= tree_nodes || right >= tree_nodes) {{
                return 0;
            }}
            if (value != value) {{
                node = ((metadata >> XGB_DEFAULT_SHIFT) & 1u) != 0u ? left : right;
            }} else {{
                node = value < split_or_leaf ? left : right;
            }}
        }}
        if (!found_leaf) {{
            return 0;
        }}
        root += tree_nodes;
    }}
    if (root != node_count) {{
        return 0;
    }}
    *result = margin;
    return 1;
}}

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    const float *features;
    const float *model;
    float *output;
    unsigned sample_count;
    unsigned feature_count;
    unsigned total_tree_count;
    unsigned magic;
    unsigned tree_count;
    unsigned node_count;
    unsigned model_feature_count;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count != XGB_CHUNK_ELEMENTS) {{
        return;
    }}
    features = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    model = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
    output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
    if (!xgb_exact_uint(features[0],
            (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / 2u, &sample_count) ||
        !xgb_exact_uint(features[1], XGB_MAX_FEATURES, &feature_count) ||
        feature_count == 0u ||
        sample_count >
            (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / (feature_count + 1u) ||
        !xgb_exact_uint(features[2], 64u, &total_tree_count) ||
        !xgb_exact_uint(model[0], XGB_MODEL_MAGIC, &magic) || magic != XGB_MODEL_MAGIC ||
        !xgb_exact_uint(model[1], 64u, &tree_count) || tree_count == 0u ||
        tree_count >= total_tree_count ||
        !xgb_exact_uint(model[2], XGB_MAX_TOTAL_NODES, &node_count) || node_count == 0u ||
        !xgb_exact_uint(model[3], XGB_MAX_FEATURES, &model_feature_count) ||
        model_feature_count != feature_count) {{
        return;
    }}
    for (unsigned index = 0u; index < XGB_CHUNK_ELEMENTS; ++index) {{
        output[index] = 0.0f;
    }}
    output[0] = (float)sample_count;
    output[1] = (float)feature_count;
    output[2] = (float)tree_count;
    output[3] = (float)total_tree_count;
    unsigned feature_elements = sample_count * feature_count;
    for (unsigned index = 0u; index < feature_elements; ++index) {{
        output[XGB_FEATURE_HEADER + index] = features[XGB_FEATURE_HEADER + index];
    }}
    unsigned partial_base = XGB_FEATURE_HEADER + feature_elements;
    for (unsigned sample = 0u; sample < sample_count; ++sample) {{
        float margin;
        if (!xgb_margin(
                features,
                model,
                sample,
                feature_count,
                tree_count,
                node_count,
                &margin)) {{
            output[3] = -1.0f;
            return;
        }}
        output[partial_base + sample] = margin;
    }}
}}
""".encode(
        "utf-8"
    )
)


RUNTIME_CHAIN_KERNEL_SOURCE: bytes = (
    f"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define XGB_CHUNK_ELEMENTS {RUNTIME_CHUNK_ELEMENTS}u
#define XGB_FEATURE_HEADER {HEADER_WORDS}u
#define XGB_OUTPUT_WORDS {OUTPUT_WORDS_PER_SAMPLE}u
#define XGB_MODEL_HEADER {RUNTIME_MODEL_HEADER_ELEMENTS}u
#define XGB_MODEL_MAGIC {RUNTIME_MODEL_MAGIC}u
#define XGB_MAX_FEATURES {RUNTIME_MAX_FEATURES}u
#define XGB_MAX_TREE_NODES {RUNTIME_MAX_NODES_PER_TREE}u
#define XGB_MAX_TOTAL_NODES {RUNTIME_MAX_TOTAL_NODES}u
#define XGB_FEATURE_MASK {_FEATURE_MASK}u
#define XGB_CHILD_MASK {_CHILD_MASK}u
#define XGB_LEAF_FEATURE {_LEAF_FEATURE}u
#define XGB_LEFT_SHIFT {_LEFT_SHIFT}u
#define XGB_RIGHT_SHIFT {_RIGHT_SHIFT}u
#define XGB_DEFAULT_SHIFT {_DEFAULT_SHIFT}u
#define XGB_TREE_SIZE_SHIFT {_TREE_SIZE_SHIFT}u

float expf(float value);

static int xgb_exact_uint(float value, unsigned maximum, unsigned *result) {{
    unsigned converted;
    if (!(value >= 0.0f) || value > (float)maximum) {{
        return 0;
    }}
    converted = (unsigned)value;
    if ((float)converted != value) {{
        return 0;
    }}
    *result = converted;
    return 1;
}}

static float xgb_sigmoid(float value) {{
    if (value >= 0.0f) {{
        return 1.0f / (1.0f + expf(-value));
    }}
    float exponential = expf(value);
    return exponential / (1.0f + exponential);
}}

static int xgb_chained_margin(
    const float *features,
    const float *model,
    unsigned sample,
    unsigned sample_count,
    unsigned feature_count,
    unsigned tree_count,
    unsigned node_count,
    float *result) {{
    unsigned root = 0u;
    unsigned partial_base = XGB_FEATURE_HEADER + sample_count * feature_count;
    float margin = features[partial_base + sample];
    if (margin != margin || margin - margin != 0.0f) {{
        return 0;
    }}
    for (unsigned tree = 0u; tree < tree_count; ++tree) {{
        unsigned root_metadata;
        unsigned tree_nodes;
        unsigned node = 0u;
        int found_leaf = 0;
        if (root >= node_count || !xgb_exact_uint(
                model[XGB_MODEL_HEADER + root * 2u], 0x7ffffu, &root_metadata)) {{
            return 0;
        }}
        tree_nodes = ((root_metadata >> XGB_TREE_SIZE_SHIFT) & XGB_CHILD_MASK) + 1u;
        if (root + tree_nodes > node_count) {{
            return 0;
        }}
        for (unsigned step = 0u; step < XGB_MAX_TREE_NODES; ++step) {{
            unsigned metadata;
            unsigned feature;
            unsigned word;
            float split_or_leaf;
            if (node >= tree_nodes) {{
                return 0;
            }}
            word = XGB_MODEL_HEADER + (root + node) * 2u;
            if (!xgb_exact_uint(model[word], 0x7ffffu, &metadata)) {{
                return 0;
            }}
            feature = metadata & XGB_FEATURE_MASK;
            split_or_leaf = model[word + 1u];
            if (feature == XGB_LEAF_FEATURE) {{
                margin += split_or_leaf;
                found_leaf = 1;
                break;
            }}
            if (feature >= feature_count) {{
                return 0;
            }}
            unsigned left = (metadata >> XGB_LEFT_SHIFT) & XGB_CHILD_MASK;
            unsigned right = (metadata >> XGB_RIGHT_SHIFT) & XGB_CHILD_MASK;
            float value = features[
                XGB_FEATURE_HEADER + sample * feature_count + feature
            ];
            if (left >= tree_nodes || right >= tree_nodes) {{
                return 0;
            }}
            if (value != value) {{
                node = ((metadata >> XGB_DEFAULT_SHIFT) & 1u) != 0u ? left : right;
            }} else {{
                node = value < split_or_leaf ? left : right;
            }}
        }}
        if (!found_leaf) {{
            return 0;
        }}
        root += tree_nodes;
    }}
    if (root != node_count) {{
        return 0;
    }}
    *result = margin;
    return 1;
}}

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    const float *features;
    const float *model;
    float *output;
    unsigned sample_count;
    unsigned feature_count;
    unsigned magic;
    unsigned first_tree_count;
    unsigned total_tree_count;
    unsigned tree_count;
    unsigned node_count;
    unsigned model_feature_count;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count != XGB_CHUNK_ELEMENTS) {{
        return;
    }}
    features = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    model = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
    output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
    if (!xgb_exact_uint(features[0],
            (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / 2u, &sample_count) ||
        !xgb_exact_uint(features[1], XGB_MAX_FEATURES, &feature_count) ||
        feature_count == 0u ||
        sample_count >
            (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / (feature_count + 1u) ||
        sample_count > (XGB_CHUNK_ELEMENTS - XGB_FEATURE_HEADER) / XGB_OUTPUT_WORDS ||
        !xgb_exact_uint(features[2], 64u, &first_tree_count) ||
        first_tree_count == 0u ||
        !xgb_exact_uint(features[3], 64u, &total_tree_count) ||
        !xgb_exact_uint(model[0], XGB_MODEL_MAGIC, &magic) || magic != XGB_MODEL_MAGIC ||
        !xgb_exact_uint(model[1], 64u, &tree_count) || tree_count == 0u ||
        first_tree_count + tree_count != total_tree_count ||
        !xgb_exact_uint(model[2], XGB_MAX_TOTAL_NODES, &node_count) || node_count == 0u ||
        !xgb_exact_uint(model[3], XGB_MAX_FEATURES, &model_feature_count) ||
        model_feature_count != feature_count || model[4] != 0.0f) {{
        return;
    }}
    for (unsigned index = 0u; index < XGB_CHUNK_ELEMENTS; ++index) {{
        output[index] = 0.0f;
    }}
    output[0] = (float)sample_count;
    output[1] = (float)feature_count;
    output[2] = (float)XGB_OUTPUT_WORDS;
    output[3] = (float)total_tree_count;
    for (unsigned sample = 0u; sample < sample_count; ++sample) {{
        float margin;
        float *record = output + XGB_FEATURE_HEADER + sample * XGB_OUTPUT_WORDS;
        if (!xgb_chained_margin(
                features,
                model,
                sample,
                sample_count,
                feature_count,
                tree_count,
                node_count,
                &margin)) {{
            output[3] = -1.0f;
            return;
        }}
        float probability = xgb_sigmoid(margin);
        record[0] = margin;
        record[1] = probability;
        record[2] = probability >= 0.5f ? 1.0f : 0.0f;
    }}
}}
""".encode(
        "utf-8"
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dependency_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in _MOVITOOLS_DEPENDENCIES:
        path = root / Path(relative)
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError(f"MoviTools dependency changed while hashing: {path}")
        records.append(
            {
                "path": relative,
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
            }
        )
    return records


def _native_runtime_identity(native_dir: str | Path | None) -> dict[str, object]:
    if native_dir is not None:
        root = Path(native_dir).expanduser().resolve()
    elif configured := os.environ.get("NPUNLOCK_NATIVE_DIR"):
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(npu.__file__).resolve().parent / "_bin"
    files: list[dict[str, object]] = []
    for relative in _NATIVE_RUNTIME_DEPENDENCIES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"native runtime dependency not found: {path}")
        stat = path.stat()
        files.append(
            {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    return {"root": str(root), "files": files}


def _provenance_identity(value: npu.IrCompileResult) -> dict[str, object]:
    return {
        "driver_index": value.driver_index,
        "device_index": value.device_index,
        "driver_version": value.driver_version,
        "vendor_id": value.vendor_id,
        "device_id": value.device_id,
        "graph_extension_version": value.graph_extension_version,
        "compiler_version": list(value.compiler_version),
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _RuntimeProgramCache:
    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        kind: str,
        sources: tuple[bytes, ...],
        movi_dll_dir: str | Path,
        native_dir: str | Path | None,
        timeout_ms: int,
    ):
        self.status = "disabled"
        self.diagnostic: str | None = None
        self.load_ms = 0.0
        self.cache_dir: Path | None = None
        self.movi_root: Path | None = None
        self.libraries: npu.NativeLibraries | None = None
        self.provenance: dict[str, object] | None = None
        self.base: dict[str, object] | None = None
        self.metadata_path: Path | None = None
        if cache_dir is None:
            return
        cache_value = os.fspath(cache_dir)
        if not cache_value:
            raise ValueError("cache_dir must not be empty")
        self.cache_dir = Path(cache_value).expanduser().resolve()
        if self.cache_dir.exists() and not self.cache_dir.is_dir():
            raise ValueError(f"cache_dir is not a directory: {self.cache_dir}")
        self.movi_root = Path(movi_dll_dir).expanduser().resolve()
        for relative in _MOVITOOLS_DEPENDENCIES:
            dependency = self.movi_root / Path(relative)
            if not dependency.is_file():
                raise FileNotFoundError(f"MoviTools dependency not found: {dependency}")
        self.libraries = npu.NativeLibraries(native_dir)
        current = self.libraries.query_ir_provenance(timeout_ms=timeout_ms)
        self.provenance = _provenance_identity(current)
        self.base = {
            "schema": _RUNTIME_CACHE_SCHEMA,
            "kind": kind,
            "sources": [hashlib.sha256(source).hexdigest() for source in sources],
            "movi_root": str(self.movi_root),
            "native_runtime": _native_runtime_identity(native_dir),
            "provenance": self.provenance,
        }
        base_key = _json_sha256(self.base)
        self.metadata_path = self.cache_dir / f"{kind}-{base_key}.json"
        self.status = "miss"

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def _dependencies_match(self, values: object) -> bool:
        if not isinstance(values, list) or len(values) != len(_MOVITOOLS_DEPENDENCIES):
            return False
        assert self.movi_root is not None
        for relative, value in zip(_MOVITOOLS_DEPENDENCIES, values):
            if not isinstance(value, dict) or set(value) != {"path", "size", "mtime_ns", "sha256"}:
                return False
            if value["path"] != relative:
                return False
            size = value["size"]
            mtime_ns = value["mtime_ns"]
            digest = value["sha256"]
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or isinstance(mtime_ns, bool)
                or not isinstance(mtime_ns, int)
                or mtime_ns < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                return False
            stat = (self.movi_root / Path(relative)).stat()
            if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
                return False
        return True

    def load(self) -> npu.Program | None:
        if not self.enabled:
            return None
        assert self.cache_dir is not None
        assert self.metadata_path is not None
        assert self.base is not None
        assert self.libraries is not None
        assert self.provenance is not None
        if not self.metadata_path.is_file():
            return None
        started = perf_counter()
        program: npu.Program | None = None
        try:
            if self.metadata_path.stat().st_size > 1024 * 1024:
                raise ValueError("cache metadata exceeds 1 MiB")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or set(metadata) != {
                "schema",
                "base",
                "dependencies",
                "full_key",
                "program",
            }:
                raise ValueError("cache metadata fields are invalid")
            if metadata["schema"] != _RUNTIME_CACHE_SCHEMA or metadata["base"] != self.base:
                raise ValueError("cache metadata identity is stale")
            dependencies = metadata["dependencies"]
            if not self._dependencies_match(dependencies):
                raise ValueError("cache dependency identity is stale")
            full_key = _json_sha256({"base": self.base, "dependencies": dependencies})
            if metadata["full_key"] != full_key:
                raise ValueError("cache full key is invalid")
            program_entry = metadata["program"]
            if not isinstance(program_entry, dict) or set(program_entry) != {"path", "size", "sha256"}:
                raise ValueError("cache program entry is invalid")
            program_hash = program_entry["sha256"]
            if (
                not isinstance(program_hash, str)
                or len(program_hash) != 64
                or any(character not in "0123456789abcdef" for character in program_hash)
            ):
                raise ValueError("cache program SHA-256 is invalid")
            expected_name = f"program-{program_hash}.npu"
            if program_entry["path"] != expected_name:
                raise ValueError("cache program path is invalid")
            program_path = self.cache_dir / expected_name
            if not program_path.is_file():
                raise ValueError("cached program is missing")
            size = program_entry["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("cached program size is invalid")
            if program_path.stat().st_size != size or _sha256_file(program_path) != program_hash:
                raise ValueError("cached program size or SHA-256 is invalid")
            program = npu.load(program_path, libraries=self.libraries)
            if program.artifact_manifest is None:
                raise ValueError("cached program has no artifact manifest")
            manifest = json.loads(program.artifact_manifest.decode("utf-8"))
            if manifest.get("provenance") != self.provenance:
                raise ValueError("cached program driver/compiler provenance is stale")
            self.status = "hit"
            self.diagnostic = None
            return program
        except Exception as error:
            if program is not None:
                program.close()
            self.status = "invalid"
            self.diagnostic = str(error)
            return None
        finally:
            self.load_ms = (perf_counter() - started) * 1000.0

    def start_dependency_hashes(self) -> tuple[ThreadPoolExecutor, Future[list[dict[str, object]]]] | None:
        if not self.enabled:
            return None
        assert self.movi_root is not None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npunlock-xgboost-cache")
        return executor, executor.submit(_dependency_records, self.movi_root)

    def store(
        self,
        program: npu.Program,
        hash_task: tuple[ThreadPoolExecutor, Future[list[dict[str, object]]]] | None,
    ) -> None:
        if not self.enabled:
            return
        assert hash_task is not None
        assert self.cache_dir is not None
        assert self.metadata_path is not None
        assert self.base is not None
        assert self.provenance is not None
        executor, future = hash_task
        program_temp: Path | None = None
        metadata_temp: Path | None = None
        previous_status = self.status
        previous_diagnostic = self.diagnostic
        try:
            dependencies = future.result()
            if program.ir_provenance is None or _provenance_identity(program.ir_provenance) != self.provenance:
                raise RuntimeError("built program provenance changed during cache construction")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix=".program-", suffix=".tmp", dir=self.cache_dir)
            os.close(descriptor)
            program_temp = Path(temp_name)
            program.export(program_temp)
            program_size = program_temp.stat().st_size
            program_hash = _sha256_file(program_temp)
            program_name = f"program-{program_hash}.npu"
            program_path = self.cache_dir / program_name
            os.replace(program_temp, program_path)
            program_temp = None
            full_key = _json_sha256({"base": self.base, "dependencies": dependencies})
            metadata = {
                "schema": _RUNTIME_CACHE_SCHEMA,
                "base": self.base,
                "dependencies": dependencies,
                "full_key": full_key,
                "program": {
                    "path": program_name,
                    "size": program_size,
                    "sha256": program_hash,
                },
            }
            descriptor, temp_name = tempfile.mkstemp(prefix=".metadata-", suffix=".tmp", dir=self.cache_dir)
            os.close(descriptor)
            metadata_temp = Path(temp_name)
            metadata_temp.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(metadata_temp, self.metadata_path)
            metadata_temp = None
            self.status = "refreshed" if previous_status == "invalid" else "stored"
            self.diagnostic = previous_diagnostic if self.status == "refreshed" else None
        except Exception as error:
            self.status = "write-error"
            self.diagnostic = str(error)
        finally:
            executor.shutdown(wait=True)
            if program_temp is not None:
                try:
                    program_temp.unlink(missing_ok=True)
                except OSError:
                    pass
            if metadata_temp is not None:
                try:
                    metadata_temp.unlink(missing_ok=True)
                except OSError:
                    pass


def _runtime_invocation_capacity(feature_count: int) -> int:
    if not isinstance(feature_count, int) or not 0 < feature_count <= RUNTIME_MAX_FEATURES:
        raise ValueError(f"feature_count must be between 1 and {RUNTIME_MAX_FEATURES}")
    return (RUNTIME_CHUNK_ELEMENTS - HEADER_WORDS) // max(feature_count, OUTPUT_WORDS_PER_SAMPLE)


def _partitioned_runtime_invocation_capacity(feature_count: int) -> int:
    if not isinstance(feature_count, int) or not 0 < feature_count <= RUNTIME_MAX_FEATURES:
        raise ValueError(f"feature_count must be between 1 and {RUNTIME_MAX_FEATURES}")
    return (RUNTIME_CHUNK_ELEMENTS - HEADER_WORDS) // max(feature_count + 1, OUTPUT_WORDS_PER_SAMPLE)


def runtime_batch_capacity(feature_count: int) -> int:
    return RUNTIME_REPLICAS * _runtime_invocation_capacity(feature_count)


def partitioned_runtime_batch_capacity(feature_count: int) -> int:
    return RUNTIME_REPLICAS * _partitioned_runtime_invocation_capacity(feature_count)


def _replicate(chunk: np.ndarray) -> np.ndarray:
    return np.tile(chunk, RUNTIME_REPLICAS).reshape(1, RUNTIME_LOGICAL_ELEMENTS)


def _shard_sizes(sample_count: int, invocation_capacity: int) -> tuple[int, ...]:
    if sample_count < 0 or invocation_capacity <= 0:
        raise ValueError("sample count and invocation capacity are invalid")
    if sample_count > RUNTIME_REPLICAS * invocation_capacity:
        raise ValueError("sample count exceeds the four-invocation capacity")
    active_count = min(RUNTIME_REPLICAS, max(1, sample_count))
    quotient, remainder = divmod(sample_count, active_count)
    sizes = tuple(quotient + (replica < remainder) for replica in range(active_count))
    if any(size > invocation_capacity for size in sizes):
        raise ValueError("balanced sample shard exceeds invocation capacity")
    return sizes


def _pack_runtime_feature_shards(
    values: np.ndarray,
    *,
    feature_count: int,
    invocation_capacity: int,
    header_tail: tuple[int, int],
) -> np.ndarray:
    chunks = np.zeros((RUNTIME_REPLICAS, RUNTIME_CHUNK_ELEMENTS), dtype=np.float32)
    sizes = _shard_sizes(values.shape[0], invocation_capacity)
    row_start = 0
    for replica, size in enumerate(sizes):
        chunk = chunks[replica]
        chunk[:HEADER_WORDS] = (size, feature_count, *header_tail)
        row_end = row_start + size
        shard = values[row_start:row_end]
        chunk[HEADER_WORDS : HEADER_WORDS + shard.size] = shard.reshape(-1)
        row_start = row_end
    for replica in range(len(sizes), RUNTIME_REPLICAS):
        chunks[replica] = chunks[0]
    return chunks.reshape(1, RUNTIME_LOGICAL_ELEMENTS)


def pack_runtime_features(
    features: object,
    *,
    feature_count: int,
    _invocation_capacity: int | None = None,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    invocation_capacity = (
        _runtime_invocation_capacity(feature_count) if _invocation_capacity is None else _invocation_capacity
    )
    capacity = RUNTIME_REPLICAS * invocation_capacity
    if values.shape[0] > capacity:
        raise ValueError(f"batch size {values.shape[0]} exceeds runtime capacity {capacity}")
    return _pack_runtime_feature_shards(
        values,
        feature_count=feature_count,
        invocation_capacity=invocation_capacity,
        header_tail=(0, 0),
    )


def pack_connected_runtime_features(
    features: object,
    *,
    feature_count: int,
    total_tree_count: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    invocation_capacity = _partitioned_runtime_invocation_capacity(feature_count)
    capacity = RUNTIME_REPLICAS * invocation_capacity
    if values.shape[0] > capacity:
        raise ValueError(f"batch size {values.shape[0]} exceeds connected runtime capacity {capacity}")
    if not 1 < total_tree_count <= 64:
        raise ValueError("connected runtime total_tree_count must be between 2 and 64")
    return _pack_runtime_feature_shards(
        values,
        feature_count=feature_count,
        invocation_capacity=invocation_capacity,
        header_tail=(total_tree_count, 0),
    )


def pack_runtime_model(ensemble: Ensemble) -> np.ndarray:
    if not 0 < ensemble.feature_count <= RUNTIME_MAX_FEATURES:
        raise ValueError(f"runtime model supports at most {RUNTIME_MAX_FEATURES} features")
    if not 0 < len(ensemble.trees) <= 64:
        raise ValueError("runtime model supports between 1 and 64 trees")
    if ensemble.node_count > RUNTIME_MAX_TOTAL_NODES:
        raise ValueError(f"runtime model has {ensemble.node_count} nodes; capacity is {RUNTIME_MAX_TOTAL_NODES}")
    if not np.isfinite(np.float32(ensemble.base_margin)):
        raise ValueError("runtime model base margin must be finite")
    chunk = np.zeros(RUNTIME_CHUNK_ELEMENTS, dtype=np.float32)
    chunk[:RUNTIME_MODEL_HEADER_ELEMENTS] = (
        RUNTIME_MODEL_MAGIC,
        len(ensemble.trees),
        ensemble.node_count,
        ensemble.feature_count,
        ensemble.base_margin,
    )
    node_base = 0
    for tree in ensemble.trees:
        if tree.node_count > RUNTIME_MAX_NODES_PER_TREE:
            raise ValueError(
                f"runtime model tree has {tree.node_count} nodes; " f"capacity is {RUNTIME_MAX_NODES_PER_TREE}"
            )
        for node in range(tree.node_count):
            if not np.isfinite(np.float32(tree.split_or_leaf[node])):
                raise ValueError("runtime model split thresholds and leaves must be finite")
            is_leaf = tree.left[node] == -1
            if is_leaf:
                metadata = _LEAF_FEATURE
            else:
                feature = tree.split_feature[node]
                if not 0 <= feature < RUNTIME_MAX_FEATURES:
                    raise ValueError("runtime model contains an out-of-range feature index")
                left = tree.left[node]
                right = tree.right[node]
                if not 0 <= left <= _CHILD_MASK or not 0 <= right <= _CHILD_MASK:
                    raise ValueError("runtime model child index does not fit the compact format")
                metadata = (
                    feature
                    | (left << _LEFT_SHIFT)
                    | (right << _RIGHT_SHIFT)
                    | (int(tree.default_left[node]) << _DEFAULT_SHIFT)
                )
            if node == 0:
                metadata |= (tree.node_count - 1) << _TREE_SIZE_SHIFT
            word = RUNTIME_MODEL_HEADER_ELEMENTS + (node_base + node) * 2
            chunk[word] = metadata
            chunk[word + 1] = tree.split_or_leaf[node]
        node_base += tree.node_count
    return _replicate(chunk)


def partition_runtime_ensemble(ensemble: Ensemble) -> tuple[Ensemble, Ensemble]:
    """Split an over-capacity ensemble at one whole-tree boundary."""
    if not 1 < len(ensemble.trees) <= 64:
        raise ValueError("partitioned runtime supports between 2 and 64 total trees")
    if ensemble.node_count <= RUNTIME_MAX_TOTAL_NODES:
        raise ValueError(
            "partitioned runtime requires a model that exceeds the single-stage "
            f"capacity of {RUNTIME_MAX_TOTAL_NODES} nodes"
        )
    if ensemble.node_count > 2 * RUNTIME_MAX_TOTAL_NODES:
        raise ValueError(
            f"partitioned runtime model has {ensemble.node_count} nodes; two-stage "
            f"capacity is {2 * RUNTIME_MAX_TOTAL_NODES}"
        )
    prefix_nodes = 0
    split_index: int | None = None
    for index, tree in enumerate(ensemble.trees[:-1], start=1):
        prefix_nodes += tree.node_count
        remaining_nodes = ensemble.node_count - prefix_nodes
        if prefix_nodes <= RUNTIME_MAX_TOTAL_NODES and remaining_nodes <= RUNTIME_MAX_TOTAL_NODES:
            split_index = index
    if split_index is None:
        raise ValueError(
            "partitioned runtime cannot split this model into two contiguous "
            f"whole-tree groups of at most {RUNTIME_MAX_TOTAL_NODES} nodes"
        )
    return (
        replace(ensemble, trees=ensemble.trees[:split_index]),
        replace(ensemble, base_margin=0.0, trees=ensemble.trees[split_index:]),
    )


def unpack_runtime_predictions(
    output: object,
    *,
    expected_samples: int,
    feature_count: int,
    tree_count: int,
    invocation_capacity: int | None = None,
) -> Predictions:
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size != RUNTIME_LOGICAL_ELEMENTS:
        raise ValueError(f"runtime kernel output has {flat.size} elements, expected {RUNTIME_LOGICAL_ELEMENTS}")
    per_invocation = _runtime_invocation_capacity(feature_count) if invocation_capacity is None else invocation_capacity
    sizes = _shard_sizes(expected_samples, per_invocation)
    chunks = flat.reshape(RUNTIME_REPLICAS, RUNTIME_CHUNK_ELEMENTS)
    record_shards: list[np.ndarray] = []
    for replica, size in enumerate(sizes):
        chunk = chunks[replica]
        header = tuple(int(float(value)) for value in chunk[:HEADER_WORDS])
        expected_header = (size, feature_count, OUTPUT_WORDS_PER_SAMPLE, tree_count)
        if header != expected_header:
            raise ValueError(f"runtime kernel output header {header!r} does not match {expected_header!r}")
        record_shards.append(
            chunk[HEADER_WORDS : HEADER_WORDS + size * OUTPUT_WORDS_PER_SAMPLE].reshape(size, OUTPUT_WORDS_PER_SAMPLE)
        )
    for replica in range(len(sizes), RUNTIME_REPLICAS):
        if not np.array_equal(chunks[0].view("V4"), chunks[replica].view("V4")):
            raise ValueError("inactive runtime ACT invocations produced different output bytes")
    records = np.concatenate(record_shards) if record_shards else np.empty((0, OUTPUT_WORDS_PER_SAMPLE))
    if not np.isfinite(records[:, :2]).all():
        raise ValueError("runtime kernel produced a non-finite margin or probability")
    if ((records[:, 1] < 0.0) | (records[:, 1] > 1.0)).any():
        raise ValueError("runtime kernel produced a probability outside [0, 1]")
    classes = records[:, 2].astype(np.int32)
    if not np.isin(classes, (0, 1)).all():
        raise ValueError("runtime kernel produced a non-binary class")
    return Predictions(
        margins=records[:, 0].copy(),
        probabilities=records[:, 1].copy(),
        classes=classes,
    )


def validate_runtime_partial(
    output: object,
    *,
    expected_samples: int,
    feature_count: int,
    tree_count: int,
    invocation_capacity: int | None = None,
) -> np.ndarray:
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size != RUNTIME_LOGICAL_ELEMENTS:
        raise ValueError(f"runtime partial output has {flat.size} elements, expected " f"{RUNTIME_LOGICAL_ELEMENTS}")
    per_invocation = (
        _partitioned_runtime_invocation_capacity(feature_count) if invocation_capacity is None else invocation_capacity
    )
    sizes = _shard_sizes(expected_samples, per_invocation)
    chunks = flat.reshape(RUNTIME_REPLICAS, RUNTIME_CHUNK_ELEMENTS)
    for replica, size in enumerate(sizes):
        chunk = chunks[replica]
        header = tuple(int(float(value)) for value in chunk[:HEADER_WORDS])
        expected_header = (size, feature_count, OUTPUT_WORDS_PER_SAMPLE, tree_count)
        if header != expected_header:
            raise ValueError(f"runtime partial output header {header!r} does not match {expected_header!r}")
        margins = chunk[HEADER_WORDS : HEADER_WORDS + size * OUTPUT_WORDS_PER_SAMPLE][::OUTPUT_WORDS_PER_SAMPLE]
        if not np.isfinite(margins).all():
            raise ValueError("runtime partial output contains a non-finite margin")
    for replica in range(len(sizes), RUNTIME_REPLICAS):
        if not np.array_equal(chunks[0].view("V4"), chunks[replica].view("V4")):
            raise ValueError("inactive runtime partial ACT invocations produced different output bytes")
    return flat.reshape(1, RUNTIME_LOGICAL_ELEMENTS)


def _pack_chained_runtime_features(
    features: object,
    partial_output: object,
    *,
    feature_count: int,
    first_tree_count: int,
    total_tree_count: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    invocation_capacity = _partitioned_runtime_invocation_capacity(feature_count)
    capacity = RUNTIME_REPLICAS * invocation_capacity
    if values.shape[0] > capacity:
        raise ValueError(f"batch size {values.shape[0]} exceeds partitioned runtime capacity {capacity}")
    if not 0 < first_tree_count < total_tree_count <= 64:
        raise ValueError("partitioned runtime tree counts must satisfy 0 < first < total <= 64")
    validated = validate_runtime_partial(
        partial_output,
        expected_samples=values.shape[0],
        feature_count=feature_count,
        tree_count=first_tree_count,
        invocation_capacity=invocation_capacity,
    ).reshape(RUNTIME_REPLICAS, RUNTIME_CHUNK_ELEMENTS)
    sizes = _shard_sizes(values.shape[0], invocation_capacity)
    chunks = np.zeros((RUNTIME_REPLICAS, RUNTIME_CHUNK_ELEMENTS), dtype=np.float32)
    row_start = 0
    for replica, size in enumerate(sizes):
        row_end = row_start + size
        shard = values[row_start:row_end]
        source = validated[replica]
        records = source[HEADER_WORDS : HEADER_WORDS + size * OUTPUT_WORDS_PER_SAMPLE].reshape(
            size, OUTPUT_WORDS_PER_SAMPLE
        )
        chunk = chunks[replica]
        chunk[:HEADER_WORDS] = (size, feature_count, first_tree_count, total_tree_count)
        feature_end = HEADER_WORDS + shard.size
        chunk[HEADER_WORDS:feature_end] = shard.reshape(-1)
        chunk[feature_end : feature_end + size] = records[:, 0]
        row_start = row_end
    for replica in range(len(sizes), RUNTIME_REPLICAS):
        chunks[replica] = chunks[0]
    return chunks.reshape(1, RUNTIME_LOGICAL_ELEMENTS)


def _validate_runtime_groups(prepared: npu.PreparedGraph, expected_count: int) -> None:
    if len(prepared.groups) != expected_count:
        raise RuntimeError(
            f"runtime XGBoost carrier exposed {len(prepared.groups)} ACT groups, " f"expected {expected_count}"
        )
    for index, group in enumerate(prepared.groups):
        if group.input_dtypes != ("f32", "f32") or group.output_dtype != "f32":
            raise RuntimeError(
                f"runtime XGBoost carrier group {index} dtype changed: "
                f"inputs={group.input_dtypes!r}, output={group.output_dtype!r}"
            )
        if (
            group.per_invocation_input_element_counts
            != (
                RUNTIME_CHUNK_ELEMENTS,
                RUNTIME_CHUNK_ELEMENTS,
            )
            or group.per_invocation_output_element_count != RUNTIME_CHUNK_ELEMENTS
        ):
            raise RuntimeError(
                f"runtime XGBoost carrier group {index} contract changed: "
                f"inputs={group.per_invocation_input_element_counts!r}, "
                f"output={group.per_invocation_output_element_count!r}"
            )
        if len(group.targets) != RUNTIME_REPLICAS:
            raise RuntimeError(
                f"runtime XGBoost carrier group {index} exposed {len(group.targets)} targets, "
                f"expected {RUNTIME_REPLICAS}"
            )


def _summarize_patch_report(payload: bytes) -> dict[str, object]:
    report = json.loads(payload.decode("utf-8"))
    return {
        "schema": report["schema"],
        "input_graph_sha256": report["input_graph_sha256"],
        "shave_elf_sha256": report["shave_elf_sha256"],
        "output_graph_sha256": report["output_graph_sha256"],
        "graph_contract": report["graph_contract"],
        "append": report["append"],
        "validation": report["validation"],
        "patched_target_count": len(report["targets"]),
    }


def _validate_runtime_program(
    program: npu.Program,
    *,
    input_names: tuple[str, ...],
    patch_report_count: int,
) -> None:
    expected_inputs = tuple((name, (1, RUNTIME_LOGICAL_ELEMENTS), "f32") for name in input_names)
    actual_inputs = tuple((contract.name, contract.shape, contract.dtype) for contract in program.input_contracts)
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"runtime XGBoost program input contract changed: {actual_inputs!r}")
    actual_outputs = tuple((contract.name, contract.shape, contract.dtype) for contract in program.output_contracts)
    expected_outputs = (("xgboost_output", (1, RUNTIME_LOGICAL_ELEMENTS), "f32"),)
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"runtime XGBoost program output contract changed: {actual_outputs!r}")
    if len(program.patch_reports) != patch_report_count:
        raise RuntimeError(
            f"runtime XGBoost program has {len(program.patch_reports)} patch reports, " f"expected {patch_report_count}"
        )
    reports: list[dict[str, object]] = []
    for index, payload in enumerate(program.patch_reports):
        try:
            report = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"runtime XGBoost patch report {index} is invalid") from error
        if not isinstance(report, dict):
            raise RuntimeError(f"runtime XGBoost patch report {index} is invalid")
        graph_contract = report.get("graph_contract")
        targets = report.get("targets")
        validation = report.get("validation")
        if (
            report.get("schema") != "npunlock.patchblob.v1"
            or not isinstance(graph_contract, dict)
            or graph_contract.get("compiler") != "8.3"
            or graph_contract.get("npu") != "3720"
            or not isinstance(targets, list)
            or len(targets) != RUNTIME_REPLICAS
            or not isinstance(validation, dict)
            or validation.get("section_contents_preserved") is not True
            or validation.get("mutation_set_preserved") is not True
        ):
            raise RuntimeError(f"runtime XGBoost patch report {index} contract changed")
        first_invocation = index * RUNTIME_REPLICAS
        for target_offset, target in enumerate(targets):
            if (
                not isinstance(target, dict)
                or target.get("invocation_index") != first_invocation + target_offset
                or target.get("range_index") != first_invocation + target_offset
                or target.get("input_count") != 2
                or target.get("element_count") != RUNTIME_CHUNK_ELEMENTS
                or target.get("span_bytes") != RUNTIME_CHUNK_ELEMENTS * 4
                or target.get("contract_flags") != 0x3B
            ):
                raise RuntimeError(f"runtime XGBoost patch report {index} target contract changed")
        reports.append(report)
    for previous, current in zip(reports, reports[1:]):
        if previous.get("output_graph_sha256") != current.get("input_graph_sha256"):
            raise RuntimeError("runtime XGBoost patch report graph hash chain is broken")
    if not reports or reports[-1].get("output_graph_sha256") != hashlib.sha256(program.graph_blob).hexdigest():
        raise RuntimeError("runtime XGBoost final graph hash does not match its patch report")


class RuntimeNpuXGBoost:
    """Binary XGBoost inference with a complete runtime-supplied FP32 forest."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        cache_dir: str | Path | None = None,
    ):
        self._initialize(
            model,
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
            timeout_ms=timeout_ms,
            cache_dir=cache_dir,
            kernel_source=RUNTIME_KERNEL_SOURCE,
            mode="normal",
        )

    def _initialize(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None,
        timeout_ms: int,
        cache_dir: str | Path | None,
        kernel_source: bytes,
        mode: str,
    ):
        if mode not in ("normal", "partial", "chained"):
            raise ValueError(f"unknown internal runtime mode {mode!r}")
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self._mode = mode
        started = perf_counter()
        self.packed_model = pack_runtime_model(self.ensemble)
        self.model_pack_ms = (perf_counter() - started) * 1000.0
        npu.configure(movi_dll_dir=movi_dll_dir)
        features = npu.input(
            "xgboost_features",
            shape=(1, RUNTIME_LOGICAL_ELEMENTS),
            dtype="f32",
        )
        forest = npu.input(
            "xgboost_forest",
            shape=(1, RUNTIME_LOGICAL_ELEMENTS),
            dtype="f32",
        )
        output = npu.custom(
            features,
            forest,
            source=kernel_source,
            carrier="Subtract",
            _name="xgboost_output",
        )
        graph = npu.Graph(
            [features, forest],
            [output],
            name=f"xgboost_runtime_{mode}",
        )
        cache_started = perf_counter()
        cache = _RuntimeProgramCache(
            cache_dir,
            kind=f"runtime-{mode}",
            sources=(kernel_source,),
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
            timeout_ms=timeout_ms,
        )
        self.program = cache.load()
        if self.program is not None:
            try:
                _validate_runtime_program(
                    self.program,
                    input_names=("xgboost_features", "xgboost_forest"),
                    patch_report_count=1,
                )
            except Exception as error:
                self.program.close()
                self.program = None
                cache.status = "invalid"
                cache.diagnostic = str(error)
        self.cache_load_ms = (perf_counter() - cache_started) * 1000.0 if cache.enabled else 0.0
        if self.program is None:
            hash_task = cache.start_dependency_hashes()
            try:
                prepared = npu.prepare(
                    graph,
                    native_dir=native_dir,
                    build_flags="",
                    timeout_ms=timeout_ms,
                    libraries=cache.libraries,
                )
                _validate_runtime_groups(prepared, 1)
                started = perf_counter()
                self.program = prepared.build(movi_dll_dir=movi_dll_dir)
                self.build_ms = (perf_counter() - started) * 1000.0
                _validate_runtime_program(
                    self.program,
                    input_names=("xgboost_features", "xgboost_forest"),
                    patch_report_count=1,
                )
                self.carrier_explanation = prepared.explain()
                cache.store(self.program, hash_task)
            except BaseException:
                if hash_task is not None:
                    hash_task[0].shutdown(wait=True)
                if self.program is not None:
                    self.program.close()
                raise
        else:
            self.build_ms = 0.0
            self.carrier_explanation = "loaded validated runtime program from cache"
        self.cache_status = cache.status
        self.cache_diagnostic = cache.diagnostic
        self.runtime_provenance = (
            cache.provenance
            if cache.provenance is not None
            else _provenance_identity(self.program.ir_provenance) if self.program.ir_provenance is not None else None
        )
        self.shared_inputs = self.program.shared_inputs()
        self.shared_output = self.program.shared_output()
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_forest"], self.packed_model)
        self.model_upload_ms = (perf_counter() - started) * 1000.0

    @classmethod
    def _partial_stage(
        cls,
        model: Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None,
        timeout_ms: int,
        cache_dir: str | Path | None = None,
    ) -> "RuntimeNpuXGBoost":
        stage = cls.__new__(cls)
        stage._initialize(
            model,
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
            timeout_ms=timeout_ms,
            cache_dir=cache_dir,
            kernel_source=RUNTIME_KERNEL_SOURCE,
            mode="partial",
        )
        return stage

    @classmethod
    def _chained_stage(
        cls,
        model: Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None,
        timeout_ms: int,
        cache_dir: str | Path | None = None,
    ) -> "RuntimeNpuXGBoost":
        stage = cls.__new__(cls)
        stage._initialize(
            model,
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
            timeout_ms=timeout_ms,
            cache_dir=cache_dir,
            kernel_source=RUNTIME_CHAIN_KERNEL_SOURCE,
            mode="chained",
        )
        return stage

    @property
    def patch_evidence(self) -> dict[str, object]:
        return _summarize_patch_report(self.program.patch_reports[0])

    @property
    def build_manifest(self) -> dict[str, object] | None:
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    def load_model(self, model: str | Path | Ensemble) -> None:
        if self._mode != "normal":
            raise RuntimeError("internal runtime stages do not support public model replacement")
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self._replace_model(ensemble)

    def _replace_model(self, ensemble: Ensemble) -> None:
        started = perf_counter()
        packed_model = pack_runtime_model(ensemble)
        model_pack_ms = (perf_counter() - started) * 1000.0
        self._upload_model(ensemble, packed_model, model_pack_ms)

    def _upload_model(
        self,
        ensemble: Ensemble,
        packed_model: np.ndarray,
        model_pack_ms: float,
    ) -> None:
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_forest"], packed_model)
        model_upload_ms = (perf_counter() - started) * 1000.0
        self.ensemble = ensemble
        self.packed_model = packed_model
        self.model_pack_ms = model_pack_ms
        self.model_upload_ms = model_upload_ms

    def _execute(
        self,
        values: np.ndarray,
        *,
        packed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if packed is None:
            started = perf_counter()
            packed = pack_runtime_features(values, feature_count=self.ensemble.feature_count)
            pack_ms = (perf_counter() - started) * 1000.0
        else:
            pack_ms = 0.0
            if packed.shape != (1, RUNTIME_LOGICAL_ELEMENTS) or packed.dtype != np.float32:
                raise ValueError("prepacked runtime features have the wrong shape or dtype")
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_features"], packed)
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        outputs = self.program.run(
            self.shared_inputs,
            outputs={"xgboost_output": self.shared_output},
        )
        graphinfer_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        copied = np.array(outputs["xgboost_output"], copy=True)
        download_ms = (perf_counter() - started) * 1000.0
        return copied, {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "graphinfer_wall_ms": graphinfer_wall_ms,
            "download_ms": download_ms,
        }

    def _run_partial(
        self,
        values: np.ndarray,
        *,
        packed: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self._mode != "partial":
            raise RuntimeError("this runtime stage cannot produce a partial output")
        copied, timings = self._execute(values, packed=packed)
        return (
            validate_runtime_partial(
                copied,
                expected_samples=values.shape[0],
                feature_count=self.ensemble.feature_count,
                tree_count=len(self.ensemble.trees),
                invocation_capacity=_partitioned_runtime_invocation_capacity(self.ensemble.feature_count),
            ),
            timings,
        )

    def _run_chained(
        self,
        values: np.ndarray,
        *,
        packed: np.ndarray,
        total_tree_count: int,
    ) -> tuple[Predictions, dict[str, float]]:
        if self._mode != "chained":
            raise RuntimeError("this runtime stage cannot consume chained margins")
        copied, timings = self._execute(values, packed=packed)
        return (
            unpack_runtime_predictions(
                copied,
                expected_samples=values.shape[0],
                feature_count=self.ensemble.feature_count,
                tree_count=total_tree_count,
                invocation_capacity=_partitioned_runtime_invocation_capacity(self.ensemble.feature_count),
            ),
            timings,
        )

    def run(self, features: object) -> tuple[Predictions, dict[str, float]]:
        if self._mode != "normal":
            raise RuntimeError("staged runtime programs are internal to the partitioned backend")
        values = np.asarray(features, dtype=np.float32)
        copied, timings = self._execute(values)
        predictions = unpack_runtime_predictions(
            copied,
            expected_samples=values.shape[0],
            feature_count=self.ensemble.feature_count,
            tree_count=len(self.ensemble.trees),
            invocation_capacity=_runtime_invocation_capacity(self.ensemble.feature_count),
        )
        return predictions, timings

    def close(self) -> None:
        self.shared_inputs.clear()
        self.shared_output = None
        self.program.close()

    def __enter__(self) -> "RuntimeNpuXGBoost":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class ConnectedRuntimeNpuXGBoost:
    """Two forest traversals connected inside one NPU graph execution."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        cache_dir: str | Path | None = None,
    ):
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self.partitions = partition_runtime_ensemble(self.ensemble)
        self.program: npu.Program | None = None
        self.shared_inputs: dict[str, np.ndarray] = {}
        self.shared_output: np.ndarray | None = None
        try:
            started = perf_counter()
            self.packed_models = tuple(pack_runtime_model(partition) for partition in self.partitions)
            self.model_pack_ms = (perf_counter() - started) * 1000.0
            npu.configure(movi_dll_dir=movi_dll_dir)
            features = npu.input(
                "xgboost_features",
                shape=(1, RUNTIME_LOGICAL_ELEMENTS),
                dtype="f32",
            )
            forest_a = npu.input(
                "xgboost_forest_a",
                shape=(1, RUNTIME_LOGICAL_ELEMENTS),
                dtype="f32",
            )
            forest_b = npu.input(
                "xgboost_forest_b",
                shape=(1, RUNTIME_LOGICAL_ELEMENTS),
                dtype="f32",
            )
            partial = npu.custom(
                features,
                forest_a,
                source=RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE,
                carrier="Subtract",
                _name="xgboost_partial",
            )
            output = npu.custom(
                partial,
                forest_b,
                source=RUNTIME_CHAIN_KERNEL_SOURCE,
                carrier="Subtract",
                _name="xgboost_output",
            )
            graph = npu.Graph(
                [features, forest_a, forest_b],
                [output],
                name="xgboost_runtime_connected",
            )
            cache_started = perf_counter()
            cache = _RuntimeProgramCache(
                cache_dir,
                kind="runtime-connected",
                sources=(RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE, RUNTIME_CHAIN_KERNEL_SOURCE),
                movi_dll_dir=movi_dll_dir,
                native_dir=native_dir,
                timeout_ms=timeout_ms,
            )
            self.program = cache.load()
            if self.program is not None:
                try:
                    _validate_runtime_program(
                        self.program,
                        input_names=("xgboost_features", "xgboost_forest_a", "xgboost_forest_b"),
                        patch_report_count=2,
                    )
                except Exception as error:
                    self.program.close()
                    self.program = None
                    cache.status = "invalid"
                    cache.diagnostic = str(error)
            self.cache_load_ms = (perf_counter() - cache_started) * 1000.0 if cache.enabled else 0.0
            if self.program is None:
                hash_task = cache.start_dependency_hashes()
                try:
                    prepared = npu.prepare(
                        graph,
                        native_dir=native_dir,
                        build_flags="",
                        timeout_ms=timeout_ms,
                        libraries=cache.libraries,
                    )
                    _validate_runtime_groups(prepared, 2)
                    mapping_indices = tuple(
                        mapping.group_indices if mapping.status == "automatic" else () for mapping in prepared.mappings
                    )
                    if mapping_indices != ((0,), (1,)):
                        raise RuntimeError(
                            "connected runtime carrier lost its one-to-one positional mapping: " f"{mapping_indices!r}"
                        )
                    plan = prepared.plan()
                    started = perf_counter()
                    self.program = plan.build(movi_dll_dir=movi_dll_dir)
                    self.build_ms = (perf_counter() - started) * 1000.0
                    _validate_runtime_program(
                        self.program,
                        input_names=("xgboost_features", "xgboost_forest_a", "xgboost_forest_b"),
                        patch_report_count=2,
                    )
                    self.carrier_explanation = prepared.explain()
                    self.build_explanation = plan.explain()
                    cache.store(self.program, hash_task)
                except BaseException:
                    if hash_task is not None:
                        hash_task[0].shutdown(wait=True)
                    raise
            else:
                self.build_ms = 0.0
                self.carrier_explanation = "loaded validated connected runtime program from cache"
                self.build_explanation = "loaded validated connected runtime program from cache"
            self.cache_status = cache.status
            self.cache_diagnostic = cache.diagnostic
            self.runtime_provenance = (
                cache.provenance
                if cache.provenance is not None
                else (
                    _provenance_identity(self.program.ir_provenance) if self.program.ir_provenance is not None else None
                )
            )
            self.shared_inputs = self.program.shared_inputs()
            self.shared_output = self.program.shared_output()
            started = perf_counter()
            np.copyto(self.shared_inputs["xgboost_forest_a"], self.packed_models[0])
            np.copyto(self.shared_inputs["xgboost_forest_b"], self.packed_models[1])
            self.model_upload_ms = (perf_counter() - started) * 1000.0
        except BaseException:
            self.close()
            raise

    @property
    def partition_tree_counts(self) -> tuple[int, int]:
        return (len(self.partitions[0].trees), len(self.partitions[1].trees))

    @property
    def partition_node_counts(self) -> tuple[int, int]:
        return (self.partitions[0].node_count, self.partitions[1].node_count)

    @property
    def patch_evidence(self) -> dict[str, dict[str, object]]:
        if self.program is None:
            raise RuntimeError("connected runtime is closed")
        return {
            "traversal_a": _summarize_patch_report(self.program.patch_reports[0]),
            "traversal_b": _summarize_patch_report(self.program.patch_reports[1]),
        }

    @property
    def build_manifest(self) -> dict[str, object] | None:
        if self.program is None:
            raise RuntimeError("connected runtime is closed")
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    def load_model(self, model: str | Path | Ensemble) -> None:
        if self.program is None:
            raise RuntimeError("connected runtime is closed")
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        partitions = partition_runtime_ensemble(ensemble)
        started = perf_counter()
        packed_models = tuple(pack_runtime_model(partition) for partition in partitions)
        model_pack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_forest_a"], packed_models[0])
        np.copyto(self.shared_inputs["xgboost_forest_b"], packed_models[1])
        model_upload_ms = (perf_counter() - started) * 1000.0
        self.ensemble = ensemble
        self.partitions = partitions
        self.packed_models = packed_models
        self.model_pack_ms = model_pack_ms
        self.model_upload_ms = model_upload_ms

    def run(self, features: object) -> tuple[Predictions, dict[str, float]]:
        if self.program is None or self.shared_output is None:
            raise RuntimeError("connected runtime is closed")
        values = np.asarray(features, dtype=np.float32)
        started = perf_counter()
        packed = pack_connected_runtime_features(
            values,
            feature_count=self.ensemble.feature_count,
            total_tree_count=len(self.ensemble.trees),
        )
        pack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_features"], packed)
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        outputs = self.program.run(
            self.shared_inputs,
            outputs={"xgboost_output": self.shared_output},
        )
        graphinfer_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        copied = np.array(outputs["xgboost_output"], copy=True)
        download_ms = (perf_counter() - started) * 1000.0
        predictions = unpack_runtime_predictions(
            copied,
            expected_samples=values.shape[0],
            feature_count=self.ensemble.feature_count,
            tree_count=len(self.ensemble.trees),
            invocation_capacity=_partitioned_runtime_invocation_capacity(self.ensemble.feature_count),
        )
        return predictions, {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "graphinfer_wall_ms": graphinfer_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.shared_inputs.clear()
        self.shared_output = None
        if self.program is not None:
            self.program.close()
            self.program = None

    def __enter__(self) -> "ConnectedRuntimeNpuXGBoost":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class PartitionedRuntimeNpuXGBoost:
    """Two chained NPU traversal stages for an over-capacity forest."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        cache_dir: str | Path | None = None,
    ):
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        self.partitions = partition_runtime_ensemble(self.ensemble)
        self.stages: list[RuntimeNpuXGBoost] = []
        try:
            self.stages.append(
                RuntimeNpuXGBoost._partial_stage(
                    self.partitions[0],
                    movi_dll_dir=movi_dll_dir,
                    native_dir=native_dir,
                    timeout_ms=timeout_ms,
                    cache_dir=cache_dir,
                )
            )
            self.stages.append(
                RuntimeNpuXGBoost._chained_stage(
                    self.partitions[1],
                    movi_dll_dir=movi_dll_dir,
                    native_dir=native_dir,
                    timeout_ms=timeout_ms,
                    cache_dir=cache_dir,
                )
            )
        except BaseException:
            self.close()
            raise

    @property
    def partition_tree_counts(self) -> tuple[int, int]:
        return (len(self.partitions[0].trees), len(self.partitions[1].trees))

    @property
    def partition_node_counts(self) -> tuple[int, int]:
        return (self.partitions[0].node_count, self.partitions[1].node_count)

    @property
    def patch_evidence(self) -> dict[str, dict[str, object]]:
        if len(self.stages) != 2:
            raise RuntimeError("partitioned runtime is closed")
        return {
            "traversal_a": self.stages[0].patch_evidence,
            "traversal_b": self.stages[1].patch_evidence,
        }

    def run(self, features: object) -> tuple[Predictions, dict[str, float]]:
        if len(self.stages) != 2:
            raise RuntimeError("partitioned runtime is closed")
        values = np.asarray(features, dtype=np.float32)
        capacity = partitioned_runtime_batch_capacity(self.ensemble.feature_count)
        if values.ndim != 2 or values.shape[1] != self.ensemble.feature_count:
            raise ValueError(f"features must have shape (N, {self.ensemble.feature_count})")
        if values.shape[0] > capacity:
            raise ValueError(f"batch size {values.shape[0]} exceeds partitioned runtime capacity {capacity}")
        started = perf_counter()
        packed = pack_runtime_features(
            values,
            feature_count=self.ensemble.feature_count,
            _invocation_capacity=_partitioned_runtime_invocation_capacity(self.ensemble.feature_count),
        )
        pack_ms = (perf_counter() - started) * 1000.0
        first, first_timings = self.stages[0]._run_partial(values, packed=packed)
        started = perf_counter()
        chained = _pack_chained_runtime_features(
            values,
            first,
            feature_count=self.ensemble.feature_count,
            first_tree_count=len(self.partitions[0].trees),
            total_tree_count=len(self.ensemble.trees),
        )
        chained_pack_ms = (perf_counter() - started) * 1000.0
        predictions, second_timings = self.stages[1]._run_chained(
            values,
            packed=chained,
            total_tree_count=len(self.ensemble.trees),
        )
        return predictions, {
            "pack_ms": pack_ms + chained_pack_ms,
            "initial_feature_pack_ms": pack_ms,
            "partial_margin_transfer_pack_ms": chained_pack_ms,
            "upload_ms": first_timings["upload_ms"] + second_timings["upload_ms"],
            "graphinfer_wall_ms": first_timings["graphinfer_wall_ms"] + second_timings["graphinfer_wall_ms"],
            "download_ms": first_timings["download_ms"] + second_timings["download_ms"],
            "traversal_a_graphinfer_wall_ms": first_timings["graphinfer_wall_ms"],
            "traversal_b_graphinfer_wall_ms": second_timings["graphinfer_wall_ms"],
        }

    def load_model(self, model: str | Path | Ensemble) -> None:
        if len(self.stages) != 2:
            raise RuntimeError("partitioned runtime is closed")
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        partitions = partition_runtime_ensemble(ensemble)
        packed_models: list[tuple[np.ndarray, float]] = []
        for partition in partitions:
            started = perf_counter()
            packed = pack_runtime_model(partition)
            packed_models.append((packed, (perf_counter() - started) * 1000.0))
        for stage, partition, (packed, pack_ms) in zip(self.stages, partitions, packed_models):
            stage._upload_model(partition, packed, pack_ms)
        self.ensemble = ensemble
        self.partitions = partitions

    def close(self) -> None:
        for stage in reversed(self.stages):
            stage.close()
        self.stages.clear()

    def __enter__(self) -> "PartitionedRuntimeNpuXGBoost":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
