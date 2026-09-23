"""One graph with independent FP32-unary and FP16-binary custom branches.

The compiler reorders these independent ACT groups relative to symbolic output
order. This example therefore discovers the carrier groups first and supplies
explicit targets selected by their validated arity and element width.
"""

import numpy as np
import npunlock as npu


scale_f32_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = in[i] * 1.5f + 0.25f;
    }
}
"""


mix_f16_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *a = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *b = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (__fp16)((float)a[i] * 0.75f + (float)b[i] * 0.25f);
    }
}
"""


def reference(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = x.astype(np.float32) * np.float32(1.5) + np.float32(0.25)
    mixed = (
        a.astype(np.float32) * np.float32(0.75)
        + b.astype(np.float32) * np.float32(0.25)
    ).astype(np.float16)
    return scaled, mixed


def build_graph(
    unary_targets: tuple[npu.PatchTarget, ...] | None = None,
    binary_targets: tuple[npu.PatchTarget, ...] | None = None,
) -> npu.Graph:
    shape = (1, 32)
    x = npu.input("x", shape=shape, dtype="f32")
    a = npu.input("a", shape=shape, dtype="f16")
    b = npu.input("b", shape=shape, dtype="f16")

    scaled = npu.custom(
        x,
        source=scale_f32_c,
        carrier="Abs",
        _name="scale_f32",
        _patch_targets=unary_targets,
    )
    mixed = npu.custom(
        a,
        b,
        source=mix_f16_c,
        carrier="Maximum",
        _name="weighted_mix_f16",
        _patch_targets=binary_targets,
    )

    return npu.Graph(
        inputs=[x, a, b],
        outputs=[mixed, scaled],
        name="mixed_precision_multi_custom",
    )


def select_group(
    groups: tuple[tuple[npu.PatchTarget, ...], ...],
    *,
    input_count: int,
    item_size: int,
) -> tuple[npu.PatchTarget, ...]:
    matches = tuple(
        group
        for group in groups
        if group
        and all(
            target.input_count == input_count
            and target.span_bytes == target.element_count * item_size
            for target in group
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {input_count}-input/{item_size}-byte ACT group; found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    native = npu.NativeLibraries()
    probe_graph = build_graph()
    serialized = npu.serialize_ir(probe_graph)
    carrier = native.compile_ir(
        serialized.xml,
        serialized.weights,
        build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',
    )
    groups = native.discover_patch_targets(carrier.graph_blob)
    unary_targets = select_group(groups, input_count=1, item_size=4)
    binary_targets = select_group(groups, input_count=2, item_size=2)

    program = npu.compile(
        build_graph(unary_targets, binary_targets),
        libraries=native,
    )

    shape = (1, 32)
    x_value = np.linspace(-3.0, 3.0, 32, dtype=np.float32).reshape(shape)
    a_value = np.linspace(-4.0, 3.5, 32, dtype=np.float16).reshape(shape)
    b_value = np.linspace(2.5, -3.0, 32, dtype=np.float16).reshape(shape)
    actual = program.run({"x": x_value, "a": a_value, "b": b_value})
    expected_f32, expected_f16 = reference(x_value, a_value, b_value)
    f32_error = np.max(np.abs(actual["scale_f32"] - expected_f32))
    f16_error = np.max(
        np.abs(actual["weighted_mix_f16"].astype(np.float32) - expected_f16.astype(np.float32))
    )
    print(f"FP32 unary maximum absolute error: {f32_error:g}")
    print(f"FP16 binary maximum absolute error: {f16_error:g}")


if __name__ == "__main__":
    main()
