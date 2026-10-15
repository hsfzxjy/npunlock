"""FP16 custom kernel with runtime dispatch through a local function table."""

import numpy as np
import npunlock as npu


function_table_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

typedef float (*transform_fn)(float value, unsigned index);

static __attribute__((noinline)) float transform_a(float value, unsigned index) {
    float adjustment = (index & 1u) != 0u ? 0.25f : 0.5f;
    return value * 0.5f - adjustment;
}

static __attribute__((noinline)) float transform_b(float value, unsigned index) {
    float adjustment = (index & 1u) != 0u ? 0.5f : 0.25f;
    return value * 1.5f + adjustment;
}

static __attribute__((noinline)) float transform_c(float value, unsigned index) {
    float adjustment = (index & 1u) != 0u ? 0.75f : -0.75f;
    return -value + adjustment;
}

static __attribute__((noinline)) transform_fn select_transform(transform_fn volatile const *table, unsigned index) {
    return table[index];
}

static __attribute__((noinline)) float dispatch(transform_fn function, float value, unsigned index) {
    return function(value, index);
}

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    transform_fn volatile transform_table[3];
    transform_table[0] = transform_a;
    transform_table[1] = transform_b;
    transform_table[2] = transform_c;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        transform_fn selected = select_transform(transform_table, i % 3u);
        output[i] = (__fp16)dispatch(selected, (float)input[i], i);
    }
}
"""


def _expected_by_invocation(values: np.ndarray, invocation_elements: int) -> np.ndarray:
    flat = values.astype(np.float32).reshape(-1)
    expected = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, flat.size, invocation_elements):
        chunk = flat[start : start + invocation_elements]
        for index, value in enumerate(chunk):
            adjustment: np.float32
            if index % 3 == 0:
                adjustment = np.float32(0.25 if index & 1 else 0.5)
                result = np.float32(value * np.float32(0.5) - adjustment)
            elif index % 3 == 1:
                adjustment = np.float32(0.5 if index & 1 else 0.25)
                result = np.float32(value * np.float32(1.5) + adjustment)
            else:
                adjustment = np.float32(0.75 if index & 1 else -0.75)
                result = np.float32(-value + adjustment)
            expected[start + index] = result
    return expected.astype(np.float16).reshape(values.shape)


def main() -> None:
    x = npu.input("x", shape=(1, 32), dtype="f16")
    y = npu.custom(x, source=function_table_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="function_table"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,):
        raise RuntimeError("carrier did not expose a stable unary invocation contract")
    if 32 % output_count != 0:
        raise RuntimeError("carrier invocation contract does not partition the logical tensor")
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
    print(f"function table dispatch matched NumPy for {output_count}-element chunks")


if __name__ == "__main__":
    main()
