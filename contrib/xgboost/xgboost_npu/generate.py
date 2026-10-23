from __future__ import annotations

import math
import numpy as np

from .model import Ensemble, Tree
from .packing import HEADER_WORDS, OUTPUT_WORDS_PER_SAMPLE, TransportSpec, transport_spec

MAX_TREES = 64
MAX_TOTAL_NODES = 1024
MAX_TREE_DEPTH = 16


def _literal(value: float) -> str:
    rounded = float(np.float32(value))
    if not math.isfinite(rounded):
        raise ValueError("generated C constants must be finite")
    text = format(rounded, ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "f"


def _tree_body(tree: Tree, node: int, indent: str) -> list[str]:
    left = tree.left[node]
    if left == -1:
        return [f"{indent}return {_literal(tree.split_or_leaf[node])};"]
    feature = tree.split_feature[node]
    threshold = _literal(tree.split_or_leaf[node])
    missing = left if tree.default_left[node] else tree.right[node]
    lines = [
        f"{indent}{{",
        f"{indent}  float value = (float)features[{feature}u];",
        f"{indent}  if (value != value) {{",
    ]
    lines.extend(_tree_body(tree, missing, indent + "    "))
    lines.extend([f"{indent}  }}", f"{indent}  if (value < {threshold}) {{"])
    lines.extend(_tree_body(tree, left, indent + "    "))
    lines.extend([f"{indent}  }} else {{"])
    lines.extend(_tree_body(tree, tree.right[node], indent + "    "))
    lines.extend([f"{indent}  }}", f"{indent}}}"])
    return lines


def generate_kernel_source(
    ensemble: Ensemble,
    *,
    transport: str | TransportSpec = "f16",
) -> bytes:
    spec = transport_spec(transport)
    if ensemble.feature_count > spec.chunk_elements - HEADER_WORDS:
        raise ValueError("feature vector cannot fit in one validated ACT invocation")
    if len(ensemble.trees) > MAX_TREES:
        raise ValueError(f"model has more than the generator limit of {MAX_TREES} trees")
    if ensemble.node_count > MAX_TOTAL_NODES:
        raise ValueError(f"model has more than the generator limit of {MAX_TOTAL_NODES} nodes")
    if ensemble.max_depth > MAX_TREE_DEPTH:
        raise ValueError(f"model exceeds the generator depth limit of {MAX_TREE_DEPTH}")
    model_description = (
        f"trees={len(ensemble.trees)}, nodes={ensemble.node_count}, "
        f"features={ensemble.feature_count}, max_depth={ensemble.max_depth}, "
        f"transport={spec.name}"
    )
    lines = [
        "#define MLIBM_DEFINE_LINK_COMPAT 1",
        "#include <npunlock/npu3720_kernel.h>",
        "",
        f"#define XGB_CHUNK_WORDS {spec.chunk_elements}u",
        f"#define XGB_HEADER_WORDS {HEADER_WORDS}u",
        f"#define XGB_FEATURE_COUNT {ensemble.feature_count}u",
        f"#define XGB_OUTPUT_WORDS {OUTPUT_WORDS_PER_SAMPLE}u",
        f"/* Generated npunlock XGBoost kernel: {model_description}. */",
        "",
        "float expf(float value);",
        "",
    ]
    for index, tree in enumerate(ensemble.trees):
        lines.append(
            f"static __attribute__((noinline)) float xgb_tree_{index}(const {spec.c_type} *features) {{"
        )
        lines.extend(_tree_body(tree, 0, "  "))
        lines.extend(["}", ""])
    lines.extend(
        [
            f"static float xgb_margin(const {spec.c_type} *features) {{",
            f"  float margin = {_literal(ensemble.base_margin)};",
        ]
    )
    for index in range(len(ensemble.trees)):
        lines.append(f"  margin += xgb_tree_{index}(features);")
    lines.extend(
        [
            "  return margin;",
            "}",
            "",
            "static float xgb_sigmoid(float margin) {",
            "  if (margin >= 0.0f) {",
            "    return 1.0f / (1.0f + expf(-margin));",
            "  }",
            "  {",
            "    float exponential = expf(margin);",
            "    return exponential / (1.0f + exponential);",
            "  }",
            "}",
            "",
            "void controlled_act(unsigned layerParams) {",
            "  act_abi_invocation invocation;",
            f"  const {spec.c_type} *input;",
            f"  {spec.c_type} *output;",
            "  unsigned sample_count;",
            "  float sample_count_value;",
            "  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);",
            "  if (invocation.element_count != XGB_CHUNK_WORDS) {",
            "    return;",
            "  }",
            f"  input = ACT_ABI_INPUT_PTR32(const {spec.c_type}, invocation, 0u);",
            f"  output = ACT_ABI_OUTPUT_PTR32({spec.c_type}, invocation, 1u);",
            "  sample_count_value = (float)input[0];",
            "  if (!(sample_count_value >= 0.0f) ||",
            "      sample_count_value > (float)((XGB_CHUNK_WORDS - XGB_HEADER_WORDS) / XGB_FEATURE_COUNT) ||",
            "      (float)input[1] != (float)XGB_FEATURE_COUNT) {",
            "    return;",
            "  }",
            "  sample_count = (unsigned)sample_count_value;",
            "  if ((float)sample_count != sample_count_value ||",
            "      XGB_HEADER_WORDS + sample_count * XGB_OUTPUT_WORDS > XGB_CHUNK_WORDS) {",
            "    return;",
            "  }",
            "  for (unsigned index = 0u; index < XGB_CHUNK_WORDS; ++index) {",
            f"    output[index] = ({spec.c_type})0.0f;",
            "  }",
            f"  output[0] = ({spec.c_type})(float)sample_count;",
            f"  output[1] = ({spec.c_type})(float)XGB_FEATURE_COUNT;",
            f"  output[2] = ({spec.c_type})(float)XGB_OUTPUT_WORDS;",
            f"  output[3] = ({spec.c_type}){_literal(float(len(ensemble.trees)))};",
            "  for (unsigned sample = 0u; sample < sample_count; ++sample) {",
            f"    const {spec.c_type} *features = "
            "input + XGB_HEADER_WORDS + sample * XGB_FEATURE_COUNT;",
            f"    {spec.c_type} *record = output + XGB_HEADER_WORDS + sample * XGB_OUTPUT_WORDS;",
            "    float margin = xgb_margin(features);",
            "    float probability = xgb_sigmoid(margin);",
            f"    record[0] = ({spec.c_type})margin;",
            f"    record[1] = ({spec.c_type})probability;",
            f"    record[2] = ({spec.c_type})(probability >= 0.5f ? 1.0f : 0.0f);",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")
