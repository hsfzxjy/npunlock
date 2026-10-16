"""FP32 custom kernel with an independently bound row-broadcast input."""

import numpy as np
import npunlock as npu


f32_broadcast_affine_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *data = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    const float *bias = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
    float *output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = data[i] * 3.0f + bias[i] * 5.0f + 7.0f;
    }
}
"""


def main() -> None:
    data = npu.input("data", shape=(4, 8), dtype="f32")
    bias = npu.input("bias", shape=(1, 8), dtype="f32")
    output = npu.custom(data, bias, source=f32_broadcast_affine_c, carrier="Subtract", _name="output")
    program = npu.compile(npu.Graph([data, bias], [output], name="f32_broadcast_custom"))

    data_value = np.arange(-16, 16, dtype=np.float32).reshape(4, 8)
    bias_value = np.asarray([[3.0, -2.0, 5.0, -4.0, 7.0, -6.0, 9.0, -8.0]], dtype=np.float32)
    expected = data_value * np.float32(3.0) + bias_value * np.float32(5.0) + np.float32(7.0)

    copied = program.run({"data": data_value, "bias": bias_value})["output"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    np.copyto(shared_inputs["data"], data_value)
    np.copyto(shared_inputs["bias"], bias_value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]

    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print("copied and shared FP32 row-broadcast results matched NumPy exactly")


if __name__ == "__main__":
    main()
