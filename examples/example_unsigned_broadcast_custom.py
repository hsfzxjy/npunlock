"""Chained scalar- and row-broadcast custom kernels for U16 and U32."""

import numpy as np
import npunlock as npu


def scalar_kernel(c_type: str) -> bytes:
    return f"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const {c_type} *data = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 0u);
    const {c_type} *scalar = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 1u);
    {c_type} *output = ACT_ABI_OUTPUT_PTR32({c_type}, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {{
        output[i] = ({c_type})((data[i] >> 2) + (scalar[0] >> 3) + 5u);
    }}
}}
""".encode()


def row_kernel(c_type: str) -> bytes:
    return f"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {{
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const {c_type} *data = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 0u);
    const {c_type} *row = ACT_ABI_INPUT_PTR32(const {c_type}, invocation, 1u);
    {c_type} *output = ACT_ABI_OUTPUT_PTR32({c_type}, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {{
        output[i] = ({c_type})((data[i] >> 1) + (row[i] >> 4) + 7u);
    }}
}}
""".encode()


def run(dtype: str, numpy_dtype: type[np.generic], c_type: str, maximum: int) -> None:
    shape = (4, 8)
    data = npu.input("data", shape=shape, dtype=dtype)
    scalar = npu.input("scalar", shape=(1, 1), dtype=dtype)
    row = npu.input("row", shape=(1, 8), dtype=dtype)
    scaled = npu.custom(
        data,
        scalar,
        source=scalar_kernel(c_type),
        carrier="Subtract",
        _name="scaled",
    )
    output = npu.custom(
        scaled,
        row,
        source=row_kernel(c_type),
        carrier="Subtract",
        _name="output",
    )
    program = npu.compile(npu.Graph([data, scalar, row], [output], name=f"{dtype}_broadcast_custom"))

    data_value = np.linspace(0, maximum, 32, dtype=numpy_dtype).reshape(shape)
    scalar_value = np.asarray([[maximum - 1214]], dtype=numpy_dtype)
    row_value = np.asarray(
        [[maximum, maximum // 2, maximum // 4, 4096, 1024, 255, 31, 3]],
        dtype=numpy_dtype,
    )
    scaled_expected = ((data_value >> numpy_dtype(2)) + (scalar_value[0, 0] >> numpy_dtype(3)) + numpy_dtype(5)).astype(
        numpy_dtype
    )
    expected = ((scaled_expected >> numpy_dtype(1)) + (row_value >> numpy_dtype(4)) + numpy_dtype(7)).astype(
        numpy_dtype
    )

    inputs = {"data": data_value, "scalar": scalar_value, "row": row_value}
    copied = program.run(inputs)["output"]
    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    for name, value in inputs.items():
        np.copyto(shared_inputs[name], value)
    shared = program.run(shared_inputs, outputs={"output": shared_output})["output"]

    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(shared, expected)
    print(f"{dtype}: chained scalar and row broadcasts matched all 32 values")


run("u16", np.uint16, "unsigned short", 65535)
run("u32", np.uint32, "unsigned int", 4294967295)
