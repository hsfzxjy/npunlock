import numpy as np
import npunlock as npu


i16_affine_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const short *in = ACT_ABI_INPUT_PTR32(const short, invocation, 0u);
    short *out = ACT_ABI_OUTPUT_PTR32(short, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (short)(in[i] * 3 - 11);
    }
}
"""


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="i16")
    y = npu.custom(x, source=i16_affine_c, carrier="Abs", _name="y")
    program = npu.compile(npu.Graph([x], [y], name="i16_custom_example"))

    input_value = np.asarray(
        [
            -10000,
            -8192,
            -6000,
            -4096,
            -3000,
            -2048,
            -1024,
            -511,
            -255,
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
            255,
            511,
            1024,
            2048,
            3000,
            4096,
            6000,
            8192,
        ],
        dtype=np.int16,
    ).reshape(1, -1)
    output = program.run_one(input_value)
    expected = (input_value.astype(np.int32) * 3 - 11).astype(np.int16)
    np.testing.assert_array_equal(output, expected)
    print("I16 custom affine kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
