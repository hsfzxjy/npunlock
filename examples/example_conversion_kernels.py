"""Connected FP32->FP16 and FP16->FP32 custom kernels with an FP16 ACT op."""

import numpy as np
import npunlock as npu


to_f16_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (__fp16)(in[i] * 0.5f + 1.0f);
    }
}
"""


to_f32_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = (float)in[i] * 2.0f - 0.25f;
    }
}
"""


shape = (1, 16)
x = npu.input("x", shape=shape, dtype="f32")
half = npu.custom(
    x,
    source=to_f16_c,
    carrier="Convert",
    destination_type="f16",
    _dtype="f16",
    _name="to_f16",
)
absolute = npu.Abs(half)
y = npu.custom(
    absolute,
    source=to_f32_c,
    carrier="Convert",
    destination_type="f32",
    _dtype="f32",
    _name="to_f32",
)

prepared = npu.prepare(npu.Graph([x], [y], name="conversion_kernels"))
to_f16_group = prepared.find_group(input_dtypes=("f32",), output_dtype="f16")
to_f32_group = prepared.find_group(input_dtypes=("f16",), output_dtype="f32")
program = prepared.build(bindings={half: to_f16_group, y: to_f32_group})

input_value = np.linspace(-5.0, 3.0, 16, dtype=np.float32).reshape(shape)
actual = program.run({"x": input_value})["to_f32"]
half_reference = (input_value * np.float32(0.5) + np.float32(1.0)).astype(np.float16)
reference = np.abs(half_reference).astype(np.float32) * np.float32(2.0) - np.float32(0.25)
np.testing.assert_allclose(actual, reference, rtol=0.0, atol=0.0)
print("connected FP32->FP16 and FP16->FP32 custom kernels matched the host oracle")
