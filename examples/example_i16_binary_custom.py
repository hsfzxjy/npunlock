import numpy as np
import npunlock as npu


i16_weighted_difference_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const short *a = ACT_ABI_INPUT_PTR32(const short, invocation, 0u);
    const short *b = ACT_ABI_INPUT_PTR32(const short, invocation, 1u);
    short *out = ACT_ABI_OUTPUT_PTR32(short, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (short)(a[i] * 3 - b[i] * 2 + 7);
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="i16")
    b = npu.input("b", shape=shape, dtype="i16")
    y = npu.custom(a, b, source=i16_weighted_difference_c, carrier="Maximum", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="i16_binary_custom_example"))

    a_value = np.asarray(
        [
            -5000,
            -4096,
            -3000,
            -2048,
            -1500,
            -1024,
            -700,
            -511,
            -300,
            -127,
            -63,
            -31,
            -15,
            -7,
            -3,
            -1,
            0,
            1,
            3,
            7,
            15,
            31,
            63,
            127,
            300,
            511,
            700,
            1024,
            1500,
            2048,
            3000,
            4096,
        ],
        dtype=np.int16,
    ).reshape(shape)
    b_value = np.asarray(
        [
            4096,
            3000,
            2048,
            1500,
            1024,
            700,
            511,
            300,
            127,
            63,
            31,
            15,
            7,
            3,
            1,
            0,
            -1,
            -3,
            -7,
            -15,
            -31,
            -63,
            -127,
            -300,
            -511,
            -700,
            -1024,
            -1500,
            -2048,
            -3000,
            -4096,
            -5000,
        ],
        dtype=np.int16,
    ).reshape(shape)
    output = program.run({"a": a_value, "b": b_value})["y"]
    expected = (a_value.astype(np.int32) * 3 - b_value.astype(np.int32) * 2 + 7).astype(np.int16)
    np.testing.assert_array_equal(output, expected)
    print("binary I16 custom kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
