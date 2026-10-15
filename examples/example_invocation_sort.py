"""FP16 insertion sort computed independently inside each ACT invocation."""

import numpy as np
import npunlock as npu


invocation_sort_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    float scratch[64];
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count > 64u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);

    for (unsigned i = 0; i < invocation.element_count; ++i) {
        scratch[i] = (float)input[i];
    }
    for (unsigned i = 1; i < invocation.element_count; ++i) {
        float key = scratch[i];
        unsigned position = i;
        while (position > 0u && scratch[position - 1u] > key) {
            scratch[position] = scratch[position - 1u];
            --position;
        }
        scratch[position] = key;
    }
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)scratch[i];
    }
}
"""


def _expected_by_invocation(values: np.ndarray, invocation_elements: int) -> np.ndarray:
    flat = values.reshape(-1)
    expected = np.empty(flat.shape, dtype=np.float16)
    for start in range(0, flat.size, invocation_elements):
        chunk = flat[start : start + invocation_elements]
        expected[start : start + chunk.size] = np.sort(chunk, kind="stable")
    return expected.reshape(values.shape)


def main() -> None:
    x = npu.input("x", shape=(1, 64), dtype="f16")
    y = npu.custom(x, source=invocation_sort_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="invocation_sort"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,):
        raise RuntimeError("carrier did not expose a stable unary invocation contract")
    logical_elements = int(np.prod(x.shape))
    if output_count > 64 or logical_elements % output_count != 0:
        raise RuntimeError("carrier invocation contract does not fit the bounded sort kernel")
    program = prepared.build()

    indices = np.arange(logical_elements, dtype=np.int16)
    values = (((indices * 17) % 67) - 33).astype(np.float16).reshape(x.shape)
    expected = _expected_by_invocation(values, output_count)

    copied = program.run({"x": values})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, values)
    shared = program.run({"x": shared_input}, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print(f"invocation-local insertion sort matched NumPy for {output_count}-element chunks")


if __name__ == "__main__":
    main()
