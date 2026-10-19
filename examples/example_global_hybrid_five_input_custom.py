"""Five-input FP16 kernel with two tensors and three graph-global parameters."""

import numpy as np
import npunlock as npu


kernel_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *data = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *residual = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    const __fp16 *scale = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 2u);
    const __fp16 *bias = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 3u);
    const __fp16 *upper = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 4u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float value = ((float)data[i] + (float)residual[i]) * (float)scale[0] + (float)bias[0];
        value = value > (float)upper[0] ? (float)upper[0] : value;
        output[i] = (__fp16)value;
    }
}
"""


def main() -> None:
    data = npu.input("data", shape=(1, 32), dtype="f16")
    residual = npu.input("residual", shape=(1, 32), dtype="f16")
    scale = npu.input("scale", shape=(1, 1), dtype="f16")
    bias = npu.input("bias", shape=(1, 1), dtype="f16")
    upper = npu.input("upper", shape=(1, 1), dtype="f16")
    output = npu.custom(
        data,
        residual,
        scale,
        bias,
        upper,
        source=kernel_c,
        carrier="FakeQuantize",
        levels=16,
        _name="output",
    )
    program = npu.compile(npu.Graph([data, residual, scale, bias, upper], [output], name="global_hybrid_five_input"))

    values = {
        "data": np.arange(-16, 16, dtype=np.float16).reshape(1, 32),
        "residual": np.arange(32, 64, dtype=np.float16).reshape(1, 32),
        "scale": np.asarray([[0.5]], dtype=np.float16),
        "bias": np.asarray([[-2]], dtype=np.float16),
        "upper": np.asarray([[32]], dtype=np.float16),
    }
    expected = np.minimum(
        (values["data"].astype(np.float32) + values["residual"].astype(np.float32)) * values["scale"].item()
        + values["bias"].item(),
        values["upper"].item(),
    ).astype(np.float16)

    copied = program.run(values)["output"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name, value in values.items():
        np.copyto(shared_inputs[name], value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]
    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print("graph-global hybrid five-input FP16 kernel matched all 32 values")


if __name__ == "__main__":
    main()
