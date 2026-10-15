"""Five-input FP16 fused RMS normalization inside each ACT invocation."""

import numpy as np
import npunlock as npu


fused_rmsnorm_c: bytes = b"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

float sqrtf(float value);

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    float scratch[64];
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count > 64u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *residual = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    const __fp16 *scale = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 2u);
    const __fp16 *bias = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 3u);
    const __fp16 *gate = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 4u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);

    float sum_of_squares = 0.0f;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float value = (float)input[i] + (float)residual[i];
        scratch[i] = value;
        sum_of_squares += value * value;
    }
    float mean_square = sum_of_squares / (float)invocation.element_count;
    float inverse_rms = 1.0f / sqrtf(mean_square + 0.25f);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float normalized = scratch[i] * inverse_rms;
        float affine = normalized * (float)scale[i] + (float)bias[i];
        output[i] = (__fp16)(affine * (float)gate[i]);
    }
}
"""


def _expected_by_invocation(values: dict[str, np.ndarray], invocation_elements: int) -> np.ndarray:
    flat = {name: value.astype(np.float32).reshape(-1) for name, value in values.items()}
    expected = np.empty(flat["input"].shape, dtype=np.float32)
    for start in range(0, flat["input"].size, invocation_elements):
        stop = start + invocation_elements
        combined = np.asarray(flat["input"][start:stop] + flat["residual"][start:stop], dtype=np.float32)
        sum_of_squares = np.float32(0.0)
        for value in combined:
            sum_of_squares = np.float32(sum_of_squares + np.float32(value * value))
        mean_square = np.float32(sum_of_squares / np.float32(invocation_elements))
        inverse_rms = np.float32(1.0) / np.sqrt(np.float32(mean_square + np.float32(0.25)))
        normalized = np.asarray(combined * inverse_rms, dtype=np.float32)
        affine = np.asarray(normalized * flat["scale"][start:stop] + flat["bias"][start:stop], dtype=np.float32)
        expected[start:stop] = np.asarray(affine * flat["gate"][start:stop], dtype=np.float32)
    return expected.astype(np.float16).reshape(values["input"].shape)


def main() -> None:
    names = ("input", "residual", "scale", "bias", "gate")
    inputs = [npu.input(name, shape=(1, 16), dtype="f16") for name in names]
    output = npu.custom(*inputs, source=fused_rmsnorm_c, carrier="FakeQuantize", levels=256, _name="output")
    prepared = npu.prepare(npu.Graph(inputs, [output], name="fused_rmsnorm"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,) * 5:
        raise RuntimeError("carrier did not expose a stable five-input invocation contract")
    if output_count > 64 or 16 % output_count != 0:
        raise RuntimeError("carrier invocation contract does not fit the bounded RMS kernel")
    program = prepared.build()

    indices = np.arange(16, dtype=np.int16)
    values = {
        "input": ((indices - 8) / 2).astype(np.float16).reshape(1, 16),
        "residual": np.where(indices & 1, 0.25, -0.25).astype(np.float16).reshape(1, 16),
        "scale": (0.75 + (indices % 4) * 0.125).astype(np.float16).reshape(1, 16),
        "bias": ((indices - 8) / 16).astype(np.float16).reshape(1, 16),
        "gate": np.where(indices & 2, 1.0, 0.5).astype(np.float16).reshape(1, 16),
    }
    expected = _expected_by_invocation(values, output_count)

    copied = program.run(values)
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name in names:
        np.copyto(shared_inputs[name], values[name])
    shared = program.run(shared_inputs, outputs={"output": shared_output})

    np.testing.assert_allclose(copied["output"], expected, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(shared["output"], expected, rtol=5e-3, atol=5e-4)
    max_error = float(np.max(np.abs(copied["output"].astype(np.float32) - expected.astype(np.float32))))
    print(
        f"fused five-input RMS normalization matched NumPy for {output_count}-element chunks; max error {max_error:.6g}"
    )


if __name__ == "__main__":
    main()
