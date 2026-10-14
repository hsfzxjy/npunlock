"""Static constant-mode U8 Pad graph with copied and shared execution."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(2, 3), dtype="u8")
    pads_begin = npu.constant(np.asarray([1, 2], dtype=np.int64), name="pads_begin")
    pads_end = npu.constant(np.asarray([1, 2], dtype=np.int64), name="pads_end")
    pad_value = npu.constant(np.asarray(255, dtype=np.uint8), name="pad_value")
    padded = npu.Pad(x, pads_begin, pads_end, pad_value, _name="padded")
    program = npu.compile(npu.Graph([x], [padded], name="u8_pad"))

    input_value = np.arange(6, dtype=np.uint8).reshape(2, 3)
    expected = np.pad(input_value, ((1, 1), (2, 2)), mode="constant", constant_values=255)

    copied = program.run({"x": input_value})
    shared_input = program.shared_input()
    shared_output = program.shared_output()
    np.copyto(shared_input, input_value)
    shared = program.run({"x": shared_input}, outputs={"padded": shared_output})

    np.testing.assert_array_equal(copied["padded"], expected)
    np.testing.assert_array_equal(shared["padded"], expected)
    print("copied and shared constant-mode U8 padding matched NumPy exactly")


if __name__ == "__main__":
    main()
