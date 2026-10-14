"""Custom FP16 kernel with an independently bound row-broadcast input."""

import numpy as np
import npunlock as npu


broadcast_add_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *data = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *bias = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)((float)data[i] + (float)bias[i]);
    }
}
"""


def main() -> None:
    data = npu.input("data", shape=(4, 8), dtype="f16")
    bias = npu.input("bias", shape=(1, 8), dtype="f16")
    output = npu.custom(data, bias, source=broadcast_add_c, carrier="Maximum", _name="output")
    program = npu.compile(npu.Graph([data, bias], [output], name="broadcast_custom"))

    data_value = np.linspace(-4.0, 3.75, 32, dtype=np.float16).reshape(4, 8)
    bias_value = np.asarray([[0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0]], dtype=np.float16)
    expected = (data_value.astype(np.float32) + bias_value.astype(np.float32)).astype(np.float16)

    copied = program.run({"data": data_value, "bias": bias_value})
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    np.copyto(shared_inputs["data"], data_value)
    np.copyto(shared_inputs["bias"], bias_value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print("copied and shared row-broadcast custom results matched NumPy exactly")


if __name__ == "__main__":
    main()
