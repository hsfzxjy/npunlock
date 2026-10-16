import numpy as np
import npunlock as npu


f32_weighted_difference_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *a = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    const float *b = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = a[i] * 3.0f - b[i] * 2.0f + 7.0f;
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="f32")
    b = npu.input("b", shape=shape, dtype="f32")
    y = npu.custom(a, b, source=f32_weighted_difference_c, carrier="Subtract", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="f32_binary_custom_example"))

    a_value = np.arange(-16, 16, dtype=np.float32).reshape(shape)
    b_value = np.arange(15, -17, -1, dtype=np.float32).reshape(shape)
    expected = a_value * np.float32(3.0) - b_value * np.float32(2.0) + np.float32(7.0)
    actual = program.run({"a": a_value, "b": b_value})["y"]

    np.testing.assert_array_equal(actual, expected)
    print(actual)
    print("all 32 FP32 outputs matched the host oracle")


if __name__ == "__main__":
    main()
