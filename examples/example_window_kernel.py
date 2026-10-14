"""FP16 five-tap window kernel with nested helpers and embedded coefficients."""

import numpy as np
import npunlock as npu


window_kernel_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

typedef struct window_context {
    const volatile float *samples;
    unsigned count;
} window_context;

const volatile float window_weights[5] = {
    0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f
};

static __attribute__((noinline)) unsigned clamp_local_index(int index, unsigned count) {
    if (index < 0) {
        return 0u;
    }
    if ((unsigned)index >= count) {
        return count - 1u;
    }
    return (unsigned)index;
}

static __attribute__((noinline)) float sample_local(const window_context *context, int index) {
    return context->samples[clamp_local_index(index, context->count)];
}

static __attribute__((noinline)) float filter_local(
    const window_context *context, unsigned index, const volatile float *weights) {
    float value = 0.0f;
    for (unsigned tap = 0; tap < 5u; ++tap) {
        int sample_index = (int)index + (int)tap - 2;
        value += sample_local(context, sample_index) * weights[tap];
    }
    return (index & 1u) != 0u ? value + 0.125f : value - 0.125f;
}

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    volatile float scratch[64];
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    if (invocation.element_count > 64u) {
        return;
    }
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        scratch[i] = (float)input[i];
    }
    const window_context context = {scratch, invocation.element_count};
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)filter_local(&context, i, window_weights);
    }
}
"""


def _expected_by_invocation(values: np.ndarray, invocation_elements: int) -> np.ndarray:
    flat = values.astype(np.float32).reshape(-1)
    expected = np.empty(flat.shape, dtype=np.float32)
    weights = np.asarray((0.0625, 0.25, 0.375, 0.25, 0.0625), dtype=np.float32)
    for start in range(0, flat.size, invocation_elements):
        chunk = flat[start : start + invocation_elements]
        for index in range(chunk.size):
            value = np.float32(0.0)
            for tap, weight in enumerate(weights):
                sample_index = min(max(index + tap - 2, 0), chunk.size - 1)
                value = np.float32(value + np.float32(chunk[sample_index] * weight))
            adjustment = np.float32(0.125 if index & 1 else -0.125)
            expected[start + index] = np.float32(value + adjustment)
    return expected.astype(np.float16).reshape(values.shape)


def main() -> None:
    x = npu.input("x", shape=(1, 64), dtype="f16")
    y = npu.custom(x, source=window_kernel_c, carrier="Abs", _name="output")
    prepared = npu.prepare(npu.Graph([x], [y], name="window_kernel"))
    group = prepared.groups[0]
    input_counts = group.per_invocation_input_element_counts
    output_count = group.per_invocation_output_element_count
    if input_counts is None or output_count is None or input_counts != (output_count,):
        raise RuntimeError("carrier did not expose a stable unary invocation contract")
    logical_elements = int(np.prod(x.shape))
    if output_count > 64 or logical_elements % output_count != 0:
        raise RuntimeError("carrier invocation contract does not fit the bounded window kernel")
    program = prepared.build()

    values = np.arange(-32, 32, dtype=np.float16).reshape(x.shape)
    expected = _expected_by_invocation(values, output_count)

    copied = program.run({"x": values})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, values)
    shared = program.run({"x": shared_input}, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print(f"five-tap window matched NumPy for {output_count}-element invocation chunks")


if __name__ == "__main__":
    main()
