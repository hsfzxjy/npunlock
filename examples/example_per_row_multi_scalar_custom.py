"""FP16 kernel with four independently bound parameters for each row."""

import numpy as np
import npunlock as npu


kernel_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *data = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *scale = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    const __fp16 *bias = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 2u);
    const __fp16 *lower = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 3u);
    const __fp16 *upper = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 4u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float value = (float)data[i] * (float)scale[0] + (float)bias[0];
        value = value < (float)lower[0] ? (float)lower[0] : value;
        value = value > (float)upper[0] ? (float)upper[0] : value;
        output[i] = (__fp16)value;
    }
}
"""


def main() -> None:
    data = npu.input("data", shape=(4, 8), dtype="f16")
    scale = npu.input("scale", shape=(4, 1), dtype="f16")
    bias = npu.input("bias", shape=(4, 1), dtype="f16")
    lower = npu.input("lower", shape=(4, 1), dtype="f16")
    upper = npu.input("upper", shape=(4, 1), dtype="f16")
    output = npu.custom(
        data,
        scale,
        bias,
        lower,
        upper,
        source=kernel_c,
        carrier="FakeQuantize",
        levels=16,
        _name="output",
    )
    program = npu.compile(npu.Graph([data, scale, bias, lower, upper], [output], name="per_row_parameters"))

    values = {
        "data": np.arange(-8, 24, dtype=np.float16).reshape(4, 8),
        "scale": np.asarray([[0.5], [1], [1.5], [2]], dtype=np.float16),
        "bias": np.asarray([[0], [1], [2], [3]], dtype=np.float16),
        "lower": np.asarray([[-2], [-1], [0], [1]], dtype=np.float16),
        "upper": np.asarray([[5], [10], [15], [20]], dtype=np.float16),
    }
    expected = np.clip(
        values["data"].astype(np.float32) * values["scale"].astype(np.float32) + values["bias"].astype(np.float32),
        values["lower"].astype(np.float32),
        values["upper"].astype(np.float32),
    ).astype(np.float16)

    copied = program.run(values)["output"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name, value in values.items():
        np.copyto(shared_inputs[name], value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]
    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print("per-row five-input FP16 kernel matched all 32 values")


if __name__ == "__main__":
    main()
