import numpy as np
import npunlock as npu


i32_weighted_difference_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const int *a = ACT_ABI_INPUT_PTR32(const int, invocation, 0u);
    const int *b = ACT_ABI_INPUT_PTR32(const int, invocation, 1u);
    int *out = ACT_ABI_OUTPUT_PTR32(int, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = a[i] * 3 - b[i] * 2 + 7;
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="i32")
    b = npu.input("b", shape=shape, dtype="i32")
    y = npu.custom(a, b, source=i32_weighted_difference_c, carrier="Minimum", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="i32_binary_custom_example"))

    a_value = np.asarray(
        [
            -100000,
            -65537,
            -32769,
            -16385,
            -8193,
            -4097,
            -2049,
            -1025,
            -513,
            -257,
            -129,
            -65,
            -33,
            -17,
            -9,
            -4,
            0,
            3,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        dtype=np.int32,
    ).reshape(shape)
    b_value = np.asarray(
        [
            70000,
            60000,
            50000,
            40000,
            30000,
            20000,
            10000,
            5000,
            2500,
            1250,
            625,
            312,
            156,
            78,
            39,
            19,
            -1,
            -5,
            -11,
            -23,
            -47,
            -95,
            -191,
            -383,
            -767,
            -1535,
            -3071,
            -6143,
            -12287,
            -24575,
            -49151,
            -98303,
        ],
        dtype=np.int32,
    ).reshape(shape)
    output = program.run({"a": a_value, "b": b_value})["y"]
    expected = (a_value.astype(np.int64) * 3 - b_value.astype(np.int64) * 2 + 7).astype(np.int32)
    np.testing.assert_array_equal(output, expected)
    print("binary I32 custom kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
