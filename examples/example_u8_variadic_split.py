"""Ordinary U8 VariadicSplit graph with unequal shared outputs."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(2, 7), dtype="u8")
    axis = npu.constant(np.asarray(1, dtype=np.int64), name="axis")
    lengths = npu.constant(np.asarray([2, 1, 4], dtype=np.int64), name="lengths")
    outputs = npu.VariadicSplit(x, axis, lengths, _name="parts")
    program = npu.compile(npu.Graph([x], list(outputs), name="u8_variadic_split"))

    input_value = np.asarray(
        [[0, 1, 2, 3, 4, 254, 255], [255, 128, 64, 32, 16, 8, 0]],
        dtype=np.uint8,
    )
    expected = np.split(input_value, [2, 3], axis=1)

    copied = program.run({"x": input_value})
    shared_input = program.shared_input()
    shared_outputs = program.shared_outputs()
    np.copyto(shared_input, input_value)
    shared = program.run({"x": shared_input}, outputs=shared_outputs)

    for index, expected_part in enumerate(expected):
        np.testing.assert_array_equal(copied[f"parts.{index}"], expected_part)
        np.testing.assert_array_equal(shared[f"parts.{index}"], expected_part)
    print("copied and shared unequal U8 splits matched exact NumPy slices")


if __name__ == "__main__":
    main()
