"""Unary FP16 custom kernel with helper calls, branches, neighbors, and scratch."""

import numpy as np
import npunlock as npu


complex_kernel_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

static __attribute__((noinline)) float branch_mix(float center, float neighbor, unsigned index) {
    if ((index & 1u) != 0u) {
        return center + neighbor * 0.5f;
    }
    return center - neighbor * 0.25f;
}

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    volatile float scratch[32];
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count > 32u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        scratch[i] = (float)input[i];
    }
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        unsigned neighbor = i + 1u < invocation.element_count ? i + 1u : i;
        output[i] = (__fp16)branch_mix(scratch[i], scratch[neighbor], i);
    }
}
"""


def _expected_by_invocation(values: np.ndarray, invocation_elements: int) -> np.ndarray:
    flat = values.astype(np.float32).reshape(-1)
    expected = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, flat.size, invocation_elements):
        chunk = flat[start : start + invocation_elements]
        neighbor = np.concatenate((chunk[1:], chunk[-1:]))
        indices = np.arange(chunk.size)
        expected[start : start + chunk.size] = np.where(
            (indices & 1) != 0,
            chunk + neighbor * 0.5,
            chunk - neighbor * 0.25,
        )
    return expected.astype(np.float16).reshape(values.shape)


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="f16")
    y = npu.custom(x, source=complex_kernel_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="complex_kernel"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,):
        raise RuntimeError("carrier did not expose a stable unary invocation contract")
    logical_elements = int(np.prod(x.shape))
    if output_count > 32 or logical_elements % output_count != 0:
        raise RuntimeError("carrier invocation contract does not fit the bounded scratch kernel")
    program = prepared.build()

    values = np.arange(-16, 16, dtype=np.float16).reshape(x.shape)
    expected = _expected_by_invocation(values, output_count)

    copied = program.run({"x": values})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, values)
    shared = program.run({"x": shared_input}, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print(f"{len(group.targets)} invocations of {output_count} elements matched NumPy with copied and shared buffers")


if __name__ == "__main__":
    main()
