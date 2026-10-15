"""FP16 local reversal driven by the observed invocation dimensions."""

import numpy as np
import npunlock as npu


dimension_aware_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.rank != 4u) {
        return;
    }
    unsigned width = act_abi_dimension(&invocation, 0u);
    if (width != invocation.element_count ||
        act_abi_dimension(&invocation, 1u) != 1u ||
        act_abi_dimension(&invocation, 2u) != 1u ||
        act_abi_dimension(&invocation, 3u) != 1u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < width; ++i) {
        output[i] = input[width - 1u - i];
    }
}
"""


def main() -> None:
    x = npu.input("x", shape=(4, 8), dtype="f16")
    y = npu.custom(x, source=dimension_aware_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="dimension_aware"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts != (8,) or output_count != 8:
        raise RuntimeError("carrier did not expose the validated eight-element invocation contract")
    program = prepared.build()

    values = np.arange(32, dtype=np.float16).reshape(x.shape)
    expected = values[:, ::-1]

    copied = program.run({"x": values})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, values)
    shared = program.run({"x": shared_input}, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print("rank-2 graph rows arrived as rank-4 (8,1,1,1) invocations and reversed exactly")


if __name__ == "__main__":
    main()
