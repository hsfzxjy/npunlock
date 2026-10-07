"""A branched graph combining DPU work and three custom ACT-SHAVE kernels."""

import numpy as np
import npunlock as npu


affine_f32_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

#ifndef SCALE_QUARTERS
#define SCALE_QUARTERS 5
#endif

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = in[i] * ((float)SCALE_QUARTERS * 0.25f) + 0.5f;
    }
}
"""


scalar_affine_f16_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *bias = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (__fp16)((float)in[i] * 0.5f + (float)bias[0]);
    }
}
"""


blend_f16_c: bytes = b"""
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


def projection_weights(size: int) -> np.ndarray:
    weights = np.zeros((size, size), dtype=np.float16)
    diagonal = np.linspace(0.5, 1.0, size, dtype=np.float16)
    weights[np.arange(size), np.arange(size)] = diagonal
    weights[np.arange(size - 1), np.arange(1, size)] = np.float16(0.125)
    return weights


def build_graph() -> tuple[npu.Graph, npu.Tensor, npu.Tensor, npu.Tensor]:
    shape = (1, 32)
    data = npu.input("data", shape=shape, dtype="f16")
    bias = npu.input("bias", shape=(1, 1), dtype="f16")
    peer = npu.input("peer", shape=shape, dtype="f16")
    control = npu.input("control", shape=shape, dtype="f32")

    positive_data = npu.Abs(data, _name="positive_data")
    biased = npu.custom(
        positive_data,
        bias,
        source=scalar_affine_f16_c,
        carrier="Maximum",
        _name="scalar_affine_f16",
    )
    weights = npu.constant(projection_weights(shape[-1]), name="projection_weights")
    projected = npu.MatMul(
        biased,
        weights,
        transpose_a=False,
        transpose_b=False,
        _shape=shape,
        _dtype="f16",
        _name="projected",
    )
    positive_peer = npu.Abs(peer, _name="positive_peer")
    blended = npu.custom(
        projected,
        positive_peer,
        source=blend_f16_c,
        carrier="Maximum",
        _name="blend_f16",
    )
    output = npu.Sqrt(blended, _name="output")

    control_output = npu.custom(
        control,
        source=affine_f32_c,
        carrier="Abs",
        _name="affine_f32",
    )
    return (
        npu.Graph(
            inputs=[data, bias, peer, control],
            outputs=[output, control_output],
            name="branched_custom_network",
        ),
        biased,
        blended,
        control_output,
    )


def reference(
    data: np.ndarray,
    bias: np.ndarray,
    peer: np.ndarray,
    control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive_data = np.abs(data).astype(np.float16)
    biased = (positive_data.astype(np.float32) * np.float32(0.5) + bias.astype(np.float32).reshape(-1)[0]).astype(
        np.float16
    )
    projected = (biased.astype(np.float32) @ projection_weights(data.shape[-1]).astype(np.float32)).astype(np.float16)
    positive_peer = np.abs(peer).astype(np.float16)
    blended = (
        projected.astype(np.float32) * np.float32(0.75) + positive_peer.astype(np.float32) * np.float32(0.25)
    ).astype(np.float16)
    output = np.sqrt(blended.astype(np.float32)).astype(np.float16)
    control_output = control * np.float32(1.25) + np.float32(0.5)
    return output, control_output


def main() -> None:
    graph, biased, blended, control_output = build_graph()
    prepared = npu.prepare(graph)
    (scalar_group,) = prepared.find_groups(
        input_count=2,
        dtype="f16",
        input_1_scalar=True,
    )
    (binary_group,) = prepared.find_groups(
        input_count=2,
        dtype="f16",
        input_1_scalar=False,
    )
    (unary_f32_group,) = prepared.find_groups(input_count=1, dtype="f32")
    print(prepared.explain())
    compiler_major, compiler_minor = prepared.ir_provenance.compiler_version
    print(
        f"device=0x{prepared.ir_provenance.device_id:04x} "
        f"driver=0x{prepared.ir_provenance.driver_version:08x} "
        f"graph compiler={compiler_major}.{compiler_minor}"
    )
    plan = prepared.plan(
        bindings={
            biased: scalar_group,
            blended: binary_group,
            control_output: unary_f32_group,
        },
        kernels={
            control_output: npu.KernelSpec(
                affine_f32_c,
                definitions=("SCALE_QUARTERS=5",),
            ),
        },
    )
    print(plan.explain())
    program = plan.build()

    shape = (1, 32)
    data = np.linspace(-2.0, 3.0, 32, dtype=np.float16).reshape(shape)
    bias = np.array([[0.75]], dtype=np.float16)
    peer = np.linspace(1.5, -2.5, 32, dtype=np.float16).reshape(shape)
    control = np.linspace(-3.0, 3.0, 32, dtype=np.float32).reshape(shape)
    actual = program.run({"data": data, "bias": bias, "peer": peer, "control": control})
    expected_output, expected_control = reference(data, bias, peer, control)
    np.testing.assert_array_equal(actual["output"], expected_output)
    np.testing.assert_array_equal(actual["affine_f32"], expected_control)
    output_error = np.max(np.abs(actual["output"].astype(np.float32) - expected_output.astype(np.float32)))
    control_error = np.max(np.abs(actual["affine_f32"] - expected_control))
    print(f"FP16 connected branch maximum absolute error: {output_error:g}")
    print(f"FP32 custom branch maximum absolute error: {control_error:g}")


if __name__ == "__main__":
    main()
