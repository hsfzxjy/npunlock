import numpy as np
import npunlock as npu


i32_affine_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const int *in = ACT_ABI_INPUT_PTR32(const int, invocation, 0u);
    int *out = ACT_ABI_OUTPUT_PTR32(int, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = in[i] * 3 + 7;
    }
}
"""


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="i32")
    y = npu.custom(x, source=i32_affine_c, carrier="Abs", _name="y")
    program = npu.compile(npu.Graph([x], [y], name="i32_custom_example"))

    input_value = np.asarray(
        [
            -1000,
            -511,
            -257,
            -128,
            -65,
            -32,
            -17,
            -9,
            -4,
            -2,
            -1,
            0,
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            34,
            55,
            89,
            144,
            233,
            377,
            610,
            987,
            1597,
            2584,
            4096,
            8191,
            16384,
        ],
        dtype=np.int32,
    ).reshape(1, -1)
    output = program.run_one(input_value)
    expected = input_value * np.int32(3) + np.int32(7)
    np.testing.assert_array_equal(output, expected)
    print("I32 custom affine kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
