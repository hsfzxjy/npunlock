import numpy as np
import npunlock as npu


i8_weighted_difference_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const signed char *a = ACT_ABI_INPUT_PTR32(const signed char, invocation, 0u);
    const signed char *b = ACT_ABI_INPUT_PTR32(const signed char, invocation, 1u);
    signed char *out = ACT_ABI_OUTPUT_PTR32(signed char, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (signed char)(a[i] * 3 - b[i] * 2 + 7);
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="i8")
    b = npu.input("b", shape=shape, dtype="i8")
    y = npu.custom(a, b, source=i8_weighted_difference_c, carrier="Subtract", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="i8_binary_custom_example"))

    a_value = np.arange(-16, 16, dtype=np.int8).reshape(shape)
    b_value = np.asarray(
        [
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -4,
            -5,
            -6,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -4,
            -5,
            -6,
            6,
            4,
            2,
            0,
            -3,
            -6,
        ],
        dtype=np.int8,
    ).reshape(shape)
    output = program.run({"a": a_value, "b": b_value})["y"]
    expected = (a_value.astype(np.int16) * 3 - b_value.astype(np.int16) * 2 + 7).astype(np.int8)
    np.testing.assert_array_equal(output, expected)
    print("binary I8 custom kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
