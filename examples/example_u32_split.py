"""Ordinary U32 Split graph with boundary values and shared buffers."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(2, 6), dtype="u32")
    axis = npu.constant(np.asarray(1, dtype=np.int64), name="axis")
    outputs = npu.Split(x, axis, num_splits=3, _name="split")
    program = npu.compile(npu.Graph([x], list(outputs), name="u32_split"))

    input_value = np.asarray(
        [
            [0, 1, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1],
            [7, 1024, 65535, 65536, 1_000_000, 4_000_000_000],
        ],
        dtype=np.uint32,
    )
    expected = np.split(input_value, 3, axis=1)
    shared_input = program.shared_input()
    shared_outputs = program.shared_outputs()
    np.copyto(shared_input, input_value)
    actual = program.run({"x": shared_input}, outputs=shared_outputs)

    for index, expected_part in enumerate(expected):
        np.testing.assert_array_equal(actual[f"split.{index}"], expected_part)
    print("three shared U32 Split outputs matched exact NumPy slices")


if __name__ == "__main__":
    main()
