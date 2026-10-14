"""Static U8 Slice graph with copied and shared execution."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(3, 8), dtype="u8")
    starts = npu.constant(np.asarray([1, 1], dtype=np.int64), name="starts")
    stops = npu.constant(np.asarray([3, 8], dtype=np.int64), name="stops")
    steps = npu.constant(np.asarray([1, 2], dtype=np.int64), name="steps")
    axes = npu.constant(np.asarray([0, 1], dtype=np.int64), name="axes")
    sliced = npu.Slice(x, starts, stops, steps, axes, _name="sliced")
    program = npu.compile(npu.Graph([x], [sliced], name="u8_slice"))

    input_value = np.arange(24, dtype=np.uint8).reshape(3, 8)
    expected = input_value[1:3, 1:8:2]

    copied = program.run({"x": input_value})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, input_value)
    shared = program.run({"x": shared_input}, outputs={"sliced": shared_output})

    np.testing.assert_array_equal(copied["sliced"], expected)
    np.testing.assert_array_equal(shared["sliced"], expected)
    print("copied and shared U8 slices matched the exact NumPy slice")


if __name__ == "__main__":
    main()
