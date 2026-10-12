"""Ordinary I16 Split graph with three outputs and shared buffers."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(2, 6), dtype="i16")
    axis = npu.constant(np.asarray(1, dtype=np.int64), name="axis")
    outputs = npu.Split(x, axis, num_splits=3, _name="split")
    program = npu.compile(npu.Graph([x], list(outputs), name="i16_split"))

    input_value = np.asarray(
        [[-32768, -12345, -1, 0, 1, 32767], [300, -300, 1024, -1024, 7, -7]],
        dtype=np.int16,
    )
    expected = np.split(input_value, 3, axis=1)
    shared_inputs = program.shared_inputs()
    shared_outputs = program.shared_outputs()
    np.copyto(shared_inputs["x"], input_value)
    actual = program.run(shared_inputs, outputs=shared_outputs)

    for index, expected_part in enumerate(expected):
        np.testing.assert_array_equal(actual[f"split.{index}"], expected_part)
    print("three shared I16 Split outputs matched exact NumPy slices")


if __name__ == "__main__":
    main()
