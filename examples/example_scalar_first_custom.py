"""Run scalar-first custom subtraction for FP16, FP32, U16, and U32."""

import numpy as np
import npunlock as npu


def kernel(c_type: str) -> bytes:
    return f"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    ACT_ABI_LOAD_OUTPUT_INVOCATION32_OR_RETURN(layerParams, 2u, invocation);
    const {c_type} *scalar = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 0u);
    const {c_type} *data = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 1u);
    {c_type} *output = ACT_ABI_OUTPUT_PTR32({c_type}, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {{
        output[i] = ({c_type})(scalar[0] - data[i]);
    }}
}}
""".encode()


def run(dtype: str, numpy_dtype: type[np.generic], c_type: str, scalar_value: object) -> None:
    scalar = npu.input("scalar", shape=(1, 1), dtype=dtype)
    data = npu.input("data", shape=(1, 32), dtype=dtype)
    output = npu.custom(
        scalar,
        data,
        source=kernel(c_type),
        carrier="Subtract",
        _shape=data.shape,
        _name="output",
    )
    program = npu.compile(npu.Graph([scalar, data], [output], name=f"{dtype}_scalar_first"))

    scalar_array = np.asarray([[scalar_value]], dtype=numpy_dtype)
    if np.issubdtype(numpy_dtype, np.floating):
        data_array = np.linspace(-7.0, 8.5, 32, dtype=numpy_dtype).reshape(1, 32)
    else:
        step = 997 if numpy_dtype == np.uint16 else 100000003
        data_array = (np.arange(32, dtype=numpy_dtype) * step).reshape(1, 32)
    expected = scalar_array - data_array
    inputs = {"scalar": scalar_array, "data": data_array}
    copied = program.run(inputs)["output"]

    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name, value in inputs.items():
        np.copyto(shared_inputs[name], value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]

    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print(f"{dtype}: scalar-first custom kernel matched all 32 values")


run("f16", np.float16, "__fp16", 17.0)
run("f32", np.float32, "float", 17.0)
run("u16", np.uint16, "unsigned short", 65535)
run("u32", np.uint32, "unsigned int", 4294967295)
