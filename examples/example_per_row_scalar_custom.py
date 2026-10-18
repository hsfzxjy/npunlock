"""Run per-row scalar custom kernels in either binary input position."""

import numpy as np
import npunlock as npu


def kernel(dtype: str, c_type: str, scalar_first: bool) -> bytes:
    load = (
        "ACT_ABI_LOAD_OUTPUT_INVOCATION32_OR_RETURN(layerParams, 2u, invocation);"
        if scalar_first
        else "ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);"
    )
    scalar_record, data_record = (0, 1) if scalar_first else (1, 0)
    if dtype in {"f16", "f32"}:
        expression = (
            f"({c_type})(scalar[0] * 0.25f - data[i] * 0.125f + 3.0f)"
            if scalar_first
            else f"({c_type})(data[i] * 0.5f + scalar[0] * 0.25f + 3.0f)"
        )
    else:
        expression = (
            f"({c_type})((scalar[0] >> 1) - (data[i] >> 3) + 7u)"
            if scalar_first
            else f"({c_type})((data[i] >> 1) + (scalar[0] >> 2) + 7u)"
        )
    return f"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    {load}
    const {c_type} *scalar = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, {scalar_record}u);
    const {c_type} *data = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, {data_record}u);
    {c_type} *output = ACT_ABI_OUTPUT_PTR32({c_type}, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {{
        output[i] = {expression};
    }}
}}
""".encode()


def run(dtype: str, numpy_dtype: type[np.generic], c_type: str, scalar_first: bool) -> None:
    row_scalar = npu.input("row_scalar", shape=(4, 1), dtype=dtype)
    data = npu.input("data", shape=(4, 8), dtype=dtype)
    inputs = (row_scalar, data) if scalar_first else (data, row_scalar)
    output = npu.custom(
        *inputs,
        source=kernel(dtype, c_type, scalar_first),
        carrier="Subtract",
        _name="output",
    )
    program = npu.compile(npu.Graph([row_scalar, data], [output], name=f"{dtype}_per_row_scalar"))

    if np.issubdtype(numpy_dtype, np.floating):
        scalar_value = np.asarray([[40], [80], [120], [160]], dtype=numpy_dtype)
        data_value = np.arange(32, dtype=numpy_dtype).reshape(4, 8) + numpy_dtype(1)
        if scalar_first:
            expected = (scalar_value.astype(np.float32) * 0.25 - data_value.astype(np.float32) * 0.125 + 3.0).astype(
                numpy_dtype
            )
        else:
            expected = (data_value.astype(np.float32) * 0.5 + scalar_value.astype(np.float32) * 0.25 + 3.0).astype(
                numpy_dtype
            )
    else:
        scalar_value = np.asarray([[30000], [35000], [40000], [45000]], dtype=numpy_dtype)
        data_value = np.arange(32, dtype=numpy_dtype).reshape(4, 8) * numpy_dtype(101) + numpy_dtype(1000)
        if scalar_first:
            expected = ((scalar_value >> numpy_dtype(1)) - (data_value >> numpy_dtype(3)) + numpy_dtype(7)).astype(
                numpy_dtype
            )
        else:
            expected = ((data_value >> numpy_dtype(1)) + (scalar_value >> numpy_dtype(2)) + numpy_dtype(7)).astype(
                numpy_dtype
            )

    values = {"row_scalar": scalar_value, "data": data_value}
    copied = program.run(values)["output"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name, value in values.items():
        np.copyto(shared_inputs[name], value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]
    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    position = "first" if scalar_first else "second"
    print(f"{dtype}: per-row scalar in the {position} input matched all 32 values")


for scalar_first in (False, True):
    run("f16", np.float16, "__fp16", scalar_first)
    run("f32", np.float32, "float", scalar_first)
    run("u16", np.uint16, "unsigned short", scalar_first)
    run("u32", np.uint32, "unsigned int", scalar_first)
