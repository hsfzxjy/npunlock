from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .model import Ensemble, load_ensemble
from .packing import HEADER_WORDS, OUTPUT_WORDS_PER_SAMPLE, Predictions


RUNTIME_CHUNK_ELEMENTS = 768
RUNTIME_LOGICAL_ELEMENTS = 3072
RUNTIME_REPLICAS = 4
RUNTIME_MODEL_HEADER_ELEMENTS = 5
RUNTIME_MODEL_MAGIC = 372020
RUNTIME_MAX_FEATURES = 63
RUNTIME_MAX_NODES_PER_TREE = 16
RUNTIME_MAX_TOTAL_NODES = (RUNTIME_CHUNK_ELEMENTS - RUNTIME_MODEL_HEADER_ELEMENTS) // 2

_FEATURE_BITS = 6
_CHILD_BITS = 4
_FEATURE_MASK = (1 << _FEATURE_BITS) - 1
_CHILD_MASK = (1 << _CHILD_BITS) - 1
_LEAF_FEATURE = _FEATURE_MASK
_LEFT_SHIFT = _FEATURE_BITS
_RIGHT_SHIFT = _LEFT_SHIFT + _CHILD_BITS
_DEFAULT_SHIFT = _RIGHT_SHIFT + _CHILD_BITS
_TREE_SIZE_SHIFT = _DEFAULT_SHIFT + 1


RUNTIME_KERNEL_SOURCE: bytes = f"""
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
""".encode("utf-8")


def runtime_batch_capacity(feature_count: int) -> int:
    if not isinstance(feature_count, int) or not 0 < feature_count <= RUNTIME_MAX_FEATURES:
        raise ValueError(f"feature_count must be between 1 and {RUNTIME_MAX_FEATURES}")
    return (RUNTIME_CHUNK_ELEMENTS - HEADER_WORDS) // feature_count


def _replicate(chunk: np.ndarray) -> np.ndarray:
    return np.tile(chunk, RUNTIME_REPLICAS).reshape(1, RUNTIME_LOGICAL_ELEMENTS)


def pack_runtime_features(features: object, *, feature_count: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    capacity = runtime_batch_capacity(feature_count)
    if values.shape[0] > capacity:
        raise ValueError(f"batch size {values.shape[0]} exceeds runtime capacity {capacity}")
    chunk = np.zeros(RUNTIME_CHUNK_ELEMENTS, dtype=np.float32)
    chunk[0] = values.shape[0]
    chunk[1] = feature_count
    chunk[HEADER_WORDS : HEADER_WORDS + values.size] = values.reshape(-1)
    return _replicate(chunk)


def pack_runtime_model(ensemble: Ensemble) -> np.ndarray:
    if not 0 < ensemble.feature_count <= RUNTIME_MAX_FEATURES:
        raise ValueError(f"runtime model supports at most {RUNTIME_MAX_FEATURES} features")
    if not 0 < len(ensemble.trees) <= 64:
        raise ValueError("runtime model supports between 1 and 64 trees")
    if ensemble.node_count > RUNTIME_MAX_TOTAL_NODES:
        raise ValueError(
            f"runtime model has {ensemble.node_count} nodes; capacity is {RUNTIME_MAX_TOTAL_NODES}"
        )
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
                f"runtime model tree has {tree.node_count} nodes; "
                f"capacity is {RUNTIME_MAX_NODES_PER_TREE}"
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


def unpack_runtime_predictions(
    output: object,
    *,
    expected_samples: int,
    feature_count: int,
    tree_count: int,
) -> Predictions:
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size != RUNTIME_LOGICAL_ELEMENTS:
        raise ValueError(
            f"runtime kernel output has {flat.size} elements, expected {RUNTIME_LOGICAL_ELEMENTS}"
        )
    first = flat[:RUNTIME_CHUNK_ELEMENTS]
    for replica in range(1, RUNTIME_REPLICAS):
        start = replica * RUNTIME_CHUNK_ELEMENTS
        other = flat[start : start + RUNTIME_CHUNK_ELEMENTS]
        if not np.array_equal(first.view("V4"), other.view("V4")):
            raise ValueError("runtime ACT invocations produced different output bytes")
    header = tuple(int(float(value)) for value in first[:HEADER_WORDS])
    expected_header = (expected_samples, feature_count, OUTPUT_WORDS_PER_SAMPLE, tree_count)
    if header != expected_header:
        raise ValueError(
            f"runtime kernel output header {header!r} does not match {expected_header!r}"
        )
    records = first[
        HEADER_WORDS : HEADER_WORDS + expected_samples * OUTPUT_WORDS_PER_SAMPLE
    ].reshape(expected_samples, OUTPUT_WORDS_PER_SAMPLE)
    classes = records[:, 2].astype(np.int32)
    if not np.isin(classes, (0, 1)).all():
        raise ValueError("runtime kernel produced a non-binary class")
    return Predictions(
        margins=records[:, 0].copy(),
        probabilities=records[:, 1].copy(),
        classes=classes,
    )


class RuntimeNpuXGBoost:
    """Binary XGBoost inference with a complete runtime-supplied FP32 forest."""

    def __init__(
        self,
        model: str | Path | Ensemble,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        self.ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
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
            source=RUNTIME_KERNEL_SOURCE,
            carrier="Subtract",
            _name="xgboost_output",
        )
        prepared = npu.prepare(
            npu.Graph([features, forest], [output], name="xgboost_runtime_forest"),
            native_dir=native_dir,
            build_flags="",
            timeout_ms=timeout_ms,
        )
        if len(prepared.groups) != 1:
            raise RuntimeError(
                f"runtime XGBoost carrier exposed {len(prepared.groups)} ACT groups, expected one"
            )
        group = prepared.groups[0]
        if group.input_dtypes != ("f32", "f32") or group.output_dtype != "f32":
            raise RuntimeError(
                "runtime XGBoost carrier dtype changed: "
                f"inputs={group.input_dtypes!r}, output={group.output_dtype!r}"
            )
        if group.per_invocation_input_element_counts != (
            RUNTIME_CHUNK_ELEMENTS,
            RUNTIME_CHUNK_ELEMENTS,
        ) or group.per_invocation_output_element_count != RUNTIME_CHUNK_ELEMENTS:
            raise RuntimeError(
                "runtime XGBoost carrier contract changed: "
                f"inputs={group.per_invocation_input_element_counts!r}, "
                f"output={group.per_invocation_output_element_count!r}"
            )
        if len(group.targets) != RUNTIME_REPLICAS:
            raise RuntimeError(
                f"runtime XGBoost carrier exposed {len(group.targets)} targets, "
                f"expected {RUNTIME_REPLICAS}"
            )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared_inputs = self.program.shared_inputs()
        self.shared_output = self.program.shared_output()
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_forest"], self.packed_model)
        self.model_upload_ms = (perf_counter() - started) * 1000.0

    @property
    def patch_evidence(self) -> dict[str, object]:
        report = json.loads(self.program.patch_reports[0].decode("utf-8"))
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

    @property
    def build_manifest(self) -> dict[str, object] | None:
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    def load_model(self, model: str | Path | Ensemble) -> None:
        ensemble = model if isinstance(model, Ensemble) else load_ensemble(model)
        started = perf_counter()
        packed_model = pack_runtime_model(ensemble)
        model_pack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        np.copyto(self.shared_inputs["xgboost_forest"], packed_model)
        model_upload_ms = (perf_counter() - started) * 1000.0
        self.ensemble = ensemble
        self.packed_model = packed_model
        self.model_pack_ms = model_pack_ms
        self.model_upload_ms = model_upload_ms

    def run(self, features: object) -> tuple[Predictions, dict[str, float]]:
        values = np.asarray(features, dtype=np.float32)
        started = perf_counter()
        packed = pack_runtime_features(values, feature_count=self.ensemble.feature_count)
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
        )
        return predictions, {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "graphinfer_wall_ms": graphinfer_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "RuntimeNpuXGBoost":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
