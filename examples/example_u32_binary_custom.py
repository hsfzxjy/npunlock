"""Equal-span two-input U32 custom kernel with unsigned arithmetic."""

import numpy as np
import npunlock as npu


u32_mix_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const unsigned int *a = ACT_ABI_INPUT_PTR32(const unsigned int, invocation, 0u);
    const unsigned int *b = ACT_ABI_INPUT_PTR32(const unsigned int, invocation, 1u);
    unsigned int *out = ACT_ABI_OUTPUT_PTR32(unsigned int, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (a[i] >> 2) + (b[i] >> 3) + 5u;
    }
}
"""


def main() -> None:
    shape = (1, 32)
    a = npu.input("a", shape=shape, dtype="u32")
    b = npu.input("b", shape=shape, dtype="u32")
    y = npu.custom(a, b, source=u32_mix_c, carrier="Subtract", _name="y")
    program = npu.compile(npu.Graph([a, b], [y], name="u32_binary_custom_example"))

    a_value = np.linspace(0, 4294967295, 32, dtype=np.uint32).reshape(shape)
    b_value = np.asarray(
        [
            4294967295,
            4294967294,
            4000000000,
            3456789012,
            3000000000,
            2147483649,
            2147483648,
            2147483647,
            1073741824,
            16777216,
            16777215,
            1048576,
            65536,
            65535,
            255,
            127,
            63,
            31,
            15,
            7,
            3,
            2,
            1,
            0,
            42,
            3141592653,
            2718281828,
            123456789,
            987654321,
            4294967000,
            1111111111,
            2222222222,
        ],
        dtype=np.uint32,
    ).reshape(shape)
    expected = (a_value >> np.uint32(2)) + (b_value >> np.uint32(3)) + np.uint32(5)

    copied = program.run({"a": a_value, "b": b_value})["y"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    np.copyto(shared_inputs["a"], a_value)
    np.copyto(shared_inputs["b"], b_value)
    shared = program.run(shared_inputs, outputs={"y": shared_output})["y"]

    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print("copied and shared U32 custom results matched all 32 host-oracle values")


if __name__ == "__main__":
    main()
