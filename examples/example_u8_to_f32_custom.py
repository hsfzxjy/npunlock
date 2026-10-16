import numpy as np
import npunlock as npu


u8_to_f32_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const unsigned char *in = ACT_ABI_INPUT_PTR32(const unsigned char, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (float)in[i] * 0.5f - 7.25f;
    }
}
"""


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="u8")
    y = npu.custom(
        x,
        source=u8_to_f32_c,
        carrier="Convert",
        destination_type="f32",
        _name="y",
    )
    prepared = npu.prepare(npu.Graph([x], [y], name="u8_to_f32_custom_example"))
    group = prepared.find_group(input_dtypes=("u8",), output_dtype="f32")
    program = prepared.build(bindings={y: group})

    input_value = np.asarray(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
            19,
            23,
            29,
            31,
            37,
            43,
            47,
            53,
            61,
            71,
            83,
            97,
            109,
            127,
            149,
            173,
            199,
            223,
            241,
            255,
        ],
        dtype=np.uint8,
    ).reshape(1, 32)
    output = program.run_one(input_value)
    expected = input_value.astype(np.float32) * np.float32(0.5) - np.float32(7.25)
    np.testing.assert_array_equal(output, expected)
    print("U8-to-FP32 custom kernel matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
