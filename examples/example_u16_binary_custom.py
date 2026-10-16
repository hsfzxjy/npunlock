"""Equal-span two-input U16 custom kernel with unsigned arithmetic."""

import numpy as np
import npunlock as npu


u16_mix_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const unsigned short *a = ACT_ABI_INPUT_PTR32(const unsigned short, invocation, 0u);
    const unsigned short *b = ACT_ABI_INPUT_PTR32(const unsigned short, invocation, 1u);
    unsigned short *out = ACT_ABI_OUTPUT_PTR32(unsigned short, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        unsigned value = ((unsigned)a[i] >> 1) + ((unsigned)b[i] >> 2) + 3u;
        out[i] = (unsigned short)value;
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="u16")
    b = npu.input("b", shape=shape, dtype="u16")
    y = npu.custom(a, b, source=u16_mix_c, carrier="Subtract", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="u16_binary_custom_example"))

    a_value = np.linspace(0, 65535, 32, dtype=np.uint16).reshape(shape)
    b_value = np.asarray(
        [
            65535,
            65534,
            60000,
            55000,
            50000,
            45000,
            40000,
            35000,
            32768,
            30000,
            25000,
            20000,
            15000,
            10000,
            5000,
            1000,
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
            1023,
            2047,
            4095,
            8191,
            16383,
            32767,
        ],
        dtype=np.uint16,
    ).reshape(shape)
    expected = (
        (a_value.astype(np.uint32) >> np.uint32(1)) + (b_value.astype(np.uint32) >> np.uint32(2)) + np.uint32(3)
    ).astype(np.uint16)

    copied = program.run({"a": a_value, "b": b_value})["y"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    np.copyto(shared_inputs["a"], a_value)
    np.copyto(shared_inputs["b"], b_value)
    shared = program.run(shared_inputs, outputs={"y": shared_output})["y"]

    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print("copied and shared U16 custom results matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
