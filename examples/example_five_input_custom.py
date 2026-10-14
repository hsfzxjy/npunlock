"""Five-input FP16 custom kernel using a FakeQuantize carrier."""

import numpy as np
import npunlock as npu


five_input_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *a = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *b = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    const __fp16 *c = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 2u);
    const __fp16 *d = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 3u);
    const __fp16 *e = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 4u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float value = (float)a[i];
        value += (float)b[i] * 2.0f;
        value += (float)c[i] * 4.0f;
        value += (float)d[i] * 8.0f;
        value += (float)e[i] * 16.0f;
        output[i] = (__fp16)value;
    }
}
"""


def main() -> None:
    names = ("a", "b", "c", "d", "e")
    inputs = [npu.input(name, shape=(1, 16), dtype="f16") for name in names]
    output = npu.custom(*inputs, source=five_input_c, carrier="FakeQuantize", levels=256, _name="output")
    program = npu.compile(npu.Graph(inputs, [output], name="five_input_custom"))

    indices = np.arange(16, dtype=np.int16)
    values = {
        "a": indices.astype(np.float16).reshape(1, 16),
        "b": (indices & 1).astype(np.float16).reshape(1, 16),
        "c": ((indices >> 1) & 1).astype(np.float16).reshape(1, 16),
        "d": ((indices >> 2) & 1).astype(np.float16).reshape(1, 16),
        "e": ((indices >> 3) & 1).astype(np.float16).reshape(1, 16),
    }
    expected = (
        values["a"].astype(np.float32)
        + values["b"].astype(np.float32) * 2.0
        + values["c"].astype(np.float32) * 4.0
        + values["d"].astype(np.float32) * 8.0
        + values["e"].astype(np.float32) * 16.0
    ).astype(np.float16)

    copied = program.run(values)
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name in names:
        np.copyto(shared_inputs[name], values[name])
    shared = program.run(shared_inputs, outputs={"output": shared_output})

    np.testing.assert_array_equal(copied["output"], expected)
    np.testing.assert_array_equal(shared["output"], expected)
    print("copied and shared five-input custom results matched NumPy exactly")


if __name__ == "__main__":
    main()
