"""FP16 softmax computed independently inside each ACT invocation."""

import numpy as np
import npunlock as npu


invocation_softmax_c: bytes = b"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

float expf(float value);

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    float scratch[64];
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count > 64u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);

    float maximum = (float)input[0];
    for (unsigned i = 1; i < invocation.element_count; ++i) {
        float value = (float)input[i];
        if (value > maximum) {
            maximum = value;
        }
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float value = expf((float)input[i] - maximum);
        scratch[i] = value;
        sum += value;
    }

    float inverse_sum = 1.0f / sum;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)(scratch[i] * inverse_sum);
    }
}
"""


def _expected_by_invocation(values: np.ndarray, invocation_elements: int) -> np.ndarray:
    flat = values.astype(np.float32).reshape(-1)
    expected = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, flat.size, invocation_elements):
        chunk = flat[start : start + invocation_elements]
        exponentials = np.exp(chunk - np.max(chunk))
        expected[start : start + chunk.size] = exponentials / np.sum(exponentials, dtype=np.float32)
    return expected.astype(np.float16).reshape(values.shape)


def main() -> None:
    x = npu.input("x", shape=(1, 64), dtype="f16")
    y = npu.custom(x, source=invocation_softmax_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="invocation_softmax"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,):
        raise RuntimeError("carrier did not expose a stable unary invocation contract")
    logical_elements = int(np.prod(x.shape))
    if output_count > 64 or logical_elements % output_count != 0:
        raise RuntimeError("carrier invocation contract does not fit the bounded softmax kernel")
    program = prepared.build()

    values = np.linspace(-2.0, 2.0, logical_elements, dtype=np.float16).reshape(x.shape)
    expected = _expected_by_invocation(values, output_count)

    copied = program.run({"x": values})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, values)
    shared = program.run({"x": shared_input}, outputs={"output": shared_output})

    np.testing.assert_allclose(copied["output"], expected, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(shared["output"], expected, rtol=5e-3, atol=5e-4)
    max_error = float(np.max(np.abs(copied["output"].astype(np.float32) - expected.astype(np.float32))))
    print(f"invocation-local softmax matched NumPy for {output_count}-element chunks; max error {max_error:.6g}")


if __name__ == "__main__":
    main()
