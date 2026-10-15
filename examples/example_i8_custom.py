import numpy as np
import npunlock as npu


i8_affine_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const signed char *in = ACT_ABI_INPUT_PTR32(const signed char, invocation, 0u);
    signed char *out = ACT_ABI_OUTPUT_PTR32(signed char, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (signed char)(in[i] * 2 + 3);
    }
}
"""


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="i8")
    y = npu.custom(x, source=i8_affine_c, carrier="Abs", _name="y")
    program = npu.compile(npu.Graph([x], [y], name="i8_custom_example"))

    input_value = np.asarray(
        [
            -60,
            -57,
            -53,
            -48,
            -43,
            -39,
            -35,
            -31,
            -27,
            -23,
            -19,
            -15,
            -11,
            -7,
            -3,
            -1,
            0,
            1,
            2,
            4,
            6,
            9,
            12,
            16,
            20,
            25,
            30,
            36,
            42,
            48,
            54,
            60,
        ],
        dtype=np.int8,
    ).reshape(1, -1)
    output = program.run_one(input_value)
    expected = (input_value.astype(np.int16) * 2 + 3).astype(np.int8)
    np.testing.assert_array_equal(output, expected)
    print("I8 custom affine kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
