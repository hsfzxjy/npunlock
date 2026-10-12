"""Ordinary U16 Split graph with boundary values and shared buffers."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(3, 4), dtype="u16")
    axis = npu.constant(np.asarray(1, dtype=np.int64), name="axis")
    outputs = npu.Split(x, axis, num_splits=2, _name="split")
    program = npu.compile(npu.Graph([x], list(outputs), name="u16_split"))

    input_value = np.asarray(
        [[0, 1, 65534, 65535], [7, 1024, 4096, 32768], [65535, 42, 0, 60000]],
        dtype=np.uint16,
    )
    expected = np.split(input_value, 2, axis=1)
    shared_input = program.shared_input()
    shared_outputs = program.shared_outputs()
    np.copyto(shared_input, input_value)
    actual = program.run({"x": shared_input}, outputs=shared_outputs)

    for index, expected_part in enumerate(expected):
        np.testing.assert_array_equal(actual[f"split.{index}"], expected_part)
    print("two shared U16 Split outputs matched exact NumPy slices")


if __name__ == "__main__":
    main()
