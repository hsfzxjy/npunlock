"""One graph with independent FP32-unary and FP16-binary custom branches.

The compiler reorders these independent ACT groups relative to symbolic output
order. A prepared graph exposes the validated groups and lets the caller bind
the symbolic custom outputs without compiling the carrier twice.
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
    mixed = (a.astype(np.float32) * np.float32(0.75) + b.astype(np.float32) * np.float32(0.25)).astype(np.float16)
    return scaled, mixed


def build_graph() -> tuple[npu.Graph, npu.Tensor, npu.Tensor]:
    shape = (1, 32)
    x = npu.input("x", shape=shape, dtype="f32")
    a = npu.input("a", shape=shape, dtype="f16")
    b = npu.input("b", shape=shape, dtype="f16")

    scaled = npu.custom(
        x,
        source=scale_f32_c,
        carrier="Abs",
        _name="scale_f32",
    )
    mixed = npu.custom(
        a,
        b,
        source=mix_f16_c,
        carrier="Maximum",
        _name="weighted_mix_f16",
    )

    return (
        npu.Graph(
            inputs=[x, a, b],
            outputs=[mixed, scaled],
            name="mixed_precision_multi_custom",
        ),
        scaled,
        mixed,
    )


def main() -> None:
    graph, scaled, mixed = build_graph()
    prepared = npu.prepare(graph)
    unary_group = prepared.find_group(input_count=1, dtype="f32")
    binary_group = prepared.find_group(input_count=2, dtype="f16")
    program = prepared.build(
        bindings={scaled: unary_group, mixed: binary_group},
    )

    shape = (1, 32)
    x_value = np.linspace(-3.0, 3.0, 32, dtype=np.float32).reshape(shape)
    a_value = np.linspace(-4.0, 3.5, 32, dtype=np.float16).reshape(shape)
    b_value = np.linspace(2.5, -3.0, 32, dtype=np.float16).reshape(shape)
    actual = program.run({"x": x_value, "a": a_value, "b": b_value})
    expected_f32, expected_f16 = reference(x_value, a_value, b_value)
    f32_error = np.max(np.abs(actual["scale_f32"] - expected_f32))
    f16_error = np.max(np.abs(actual["weighted_mix_f16"].astype(np.float32) - expected_f16.astype(np.float32)))
    print(f"FP32 unary maximum absolute error: {f32_error:g}")
    print(f"FP16 binary maximum absolute error: {f16_error:g}")


if __name__ == "__main__":
    main()
