"""Connected mixed-precision graph with three custom kernels and DPU work."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import npunlock as npu


to_f16_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (__fp16)(in[i] * 0.5f + 1.0f);
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


to_f32_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (float)in[i] * 2.0f - 0.25f;
    }
}
"""


def projection_weights(size: int) -> np.ndarray:
    weights = np.zeros((size, size), dtype=np.float16)
    weights[np.arange(size), np.arange(size)] = np.linspace(0.5, 1.0, size, dtype=np.float16)
    weights[np.arange(size - 1), np.arange(1, size)] = np.float16(0.125)
    return weights


def build_graph() -> tuple[npu.Graph, npu.Tensor, npu.Tensor, npu.Tensor]:
    shape = (1, 32)
    data = npu.input("data", shape=shape, dtype="f32")
    peer = npu.input("peer", shape=shape, dtype="f16")
    half = npu.custom(
        data,
        source=to_f16_c,
        carrier="Convert",
        destination_type="f16",
        _dtype="f16",
        _name="to_f16",
    )
    blended = npu.custom(half, peer, source=blend_f16_c, carrier="Maximum", _name="blend_f16")
    weights = npu.constant(projection_weights(shape[-1]), name="projection_weights")
    projected = npu.MatMul(
        blended,
        weights,
        transpose_a=False,
        transpose_b=False,
        _shape=shape,
        _dtype="f16",
        _name="projected",
    )
    output = npu.custom(
        projected,
        source=to_f32_c,
        carrier="Convert",
        destination_type="f32",
        _dtype="f32",
        _name="to_f32",
    )
    return npu.Graph([data, peer], [output], name="connected_mixed_precision"), half, blended, output


def reference(data: np.ndarray, peer: np.ndarray) -> np.ndarray:
    half = (data * np.float32(0.5) + np.float32(1.0)).astype(np.float16)
    blended = (half.astype(np.float32) * np.float32(0.75) + peer.astype(np.float32) * np.float32(0.25)).astype(
        np.float16
    )
    projected = (blended.astype(np.float32) @ projection_weights(data.shape[-1]).astype(np.float32)).astype(np.float16)
    return projected.astype(np.float32) * np.float32(2.0) - np.float32(0.25)


def make_plan(prepared: npu.PreparedGraph, half: npu.Tensor, blended: npu.Tensor, output: npu.Tensor) -> npu.BuildPlan:
    (to_f16_group,) = prepared.find_groups(input_dtypes=("f32",), output_dtype="f16")
    (blend_group,) = prepared.find_groups(input_dtypes=("f16", "f16"), output_dtype="f16")
    (to_f32_group,) = prepared.find_groups(input_dtypes=("f16",), output_dtype="f32")
    return prepared.plan(bindings={half: to_f16_group, blended: blend_group, output: to_f32_group})


def inputs(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    shape = (1, 32)
    data = (np.linspace(-4.0, 3.0, 32, dtype=np.float32) + np.float32(offset)).reshape(shape)
    peer = np.linspace(2.5, -1.5, 32, dtype=np.float16).reshape(shape)
    return data, peer


def main() -> None:
    graph, half, blended, output = build_graph()
    prepared = npu.prepare(graph)
    plan = make_plan(prepared, half, blended, output)
    print(prepared.explain())
    print(plan.explain())

    artifact_root = Path("build/example-connected-mixed-precision")
    artifact_root.mkdir(parents=True, exist_ok=True)
    prepared_path = artifact_root / "carrier.npuprepared"
    program_path = artifact_root / "program.npu"
    prepared.export(prepared_path, plan=plan)

    loaded_prepared = npu.load_prepared(prepared_path, graph=graph)
    loaded_plan = loaded_prepared.plan()
    with plan.build() as original, loaded_plan.build() as restored:
        for offset in (0.0, 0.375):
            data, peer = inputs(offset)
            expected = reference(data, peer)
            np.testing.assert_array_equal(original.run({"data": data, "peer": peer})["to_f32"], expected)
            np.testing.assert_array_equal(restored.run({"data": data, "peer": peer})["to_f32"], expected)
        original.export(program_path)

    with npu.load(program_path) as reloaded:
        data, peer = inputs(-0.25)
        np.testing.assert_array_equal(reloaded.run({"data": data, "peer": peer})["to_f32"], reference(data, peer))

    print("connected mixed-precision carrier, prepared reload, and saved-program reload matched exact host oracles")


if __name__ == "__main__":
    main()
