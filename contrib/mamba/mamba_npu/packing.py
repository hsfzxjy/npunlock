from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAGIC = 0x4D414D31
VERSION = 1
REPLICAS = 4
INVOCATION_WORDS = 32768
LOGICAL_WORDS = REPLICAS * INVOCATION_WORDS
HEADER_WORDS = 16
MAX_STATE = 16
MAX_SEQUENCE = 16


@dataclass(frozen=True, slots=True)
class Layout:
    channels: int
    state_size: int
    sequence_length: int
    channel_slices: tuple[slice, ...]


def make_layout(channels: int, state_size: int, sequence_length: int) -> Layout:
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be a positive integer")
    if not isinstance(state_size, int) or not 1 <= state_size <= MAX_STATE:
        raise ValueError(f"state_size must be in [1, {MAX_STATE}]")
    if not isinstance(sequence_length, int) or not 1 <= sequence_length <= MAX_SEQUENCE:
        raise ValueError(f"sequence_length must be in [1, {MAX_SEQUENCE}]")
    base, remainder = divmod(channels, REPLICAS)
    slices = []
    begin = 0
    for replica in range(REPLICAS):
        count = base + (1 if replica < remainder else 0)
        slices.append(slice(begin, begin + count))
        begin += count
    for channel_slice in slices:
        count = channel_slice.stop - channel_slice.start
        dynamic_required = HEADER_WORDS + 2 * sequence_length * state_size + count * (
            3 * sequence_length + state_size
        )
        model_required = HEADER_WORDS + count * (state_size + 2)
        output_required = HEADER_WORDS + count * (sequence_length + state_size)
        if max(dynamic_required, model_required, output_required) > INVOCATION_WORDS:
            raise ValueError("selective-scan payload exceeds the validated invocation capacity")
    return Layout(channels, state_size, sequence_length, tuple(slices))


def _headers(buffer: np.ndarray, replica: int, values: tuple[int, ...]) -> None:
    words = buffer.reshape(REPLICAS, INVOCATION_WORDS).view(np.uint32)
    words[replica, : len(values)] = values


def pack_model(a: object, d: object, delta_bias: object, *, sequence_length: int = 1) -> tuple[np.ndarray, Layout]:
    a_value = np.asarray(a, dtype=np.float32)
    d_value = np.asarray(d, dtype=np.float32)
    bias_value = np.asarray(delta_bias, dtype=np.float32)
    if a_value.ndim != 2:
        raise ValueError("a must have shape (channels, state_size)")
    layout = make_layout(a_value.shape[0], a_value.shape[1], sequence_length)
    if d_value.shape != (layout.channels,) or bias_value.shape != (layout.channels,):
        raise ValueError("d and delta_bias must have shape (channels,)")
    packed = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
    for replica, channel_slice in enumerate(layout.channel_slices):
        count = channel_slice.stop - channel_slice.start
        _headers(packed, replica, (MAGIC, VERSION, count, layout.state_size))
        cursor = HEADER_WORDS
        for channel in range(channel_slice.start, channel_slice.stop):
            packed[replica, cursor : cursor + layout.state_size] = a_value[channel]
            cursor += layout.state_size
            packed[replica, cursor] = d_value[channel]
            packed[replica, cursor + 1] = bias_value[channel]
            cursor += 2
    return packed.reshape(1, LOGICAL_WORDS), layout


def pack_dynamic(
    u: object,
    delta: object,
    b: object,
    c: object,
    z: object,
    initial_state: object | None = None,
) -> tuple[np.ndarray, Layout]:
    u_value = np.asarray(u, dtype=np.float32)
    delta_value = np.asarray(delta, dtype=np.float32)
    b_value = np.asarray(b, dtype=np.float32)
    c_value = np.asarray(c, dtype=np.float32)
    z_value = np.asarray(z, dtype=np.float32)
    if u_value.ndim != 2 or b_value.ndim != 2:
        raise ValueError("u and b must be matrices")
    layout = make_layout(u_value.shape[0], b_value.shape[1], u_value.shape[1])
    if delta_value.shape != u_value.shape or z_value.shape != u_value.shape:
        raise ValueError("delta and z must match u")
    if b_value.shape != (layout.sequence_length, layout.state_size) or c_value.shape != b_value.shape:
        raise ValueError("b and c must have shape (sequence_length, state_size)")
    if initial_state is None:
        state_value = np.zeros((layout.channels, layout.state_size), dtype=np.float32)
    else:
        state_value = np.asarray(initial_state, dtype=np.float32)
        if state_value.shape != (layout.channels, layout.state_size):
            raise ValueError("initial_state has the wrong shape")
    packed = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
    for replica, channel_slice in enumerate(layout.channel_slices):
        count = channel_slice.stop - channel_slice.start
        _headers(
            packed,
            replica,
            (MAGIC, VERSION, count, layout.state_size, layout.sequence_length),
        )
        cursor = HEADER_WORDS
        packed[replica, cursor : cursor + b_value.size] = b_value.reshape(-1)
        cursor += b_value.size
        packed[replica, cursor : cursor + c_value.size] = c_value.reshape(-1)
        cursor += c_value.size
        for channel in range(channel_slice.start, channel_slice.stop):
            packed[replica, cursor : cursor + layout.sequence_length] = u_value[channel]
            cursor += layout.sequence_length
            packed[replica, cursor : cursor + layout.sequence_length] = delta_value[channel]
            cursor += layout.sequence_length
            packed[replica, cursor : cursor + layout.sequence_length] = z_value[channel]
            cursor += layout.sequence_length
            packed[replica, cursor : cursor + layout.state_size] = state_value[channel]
            cursor += layout.state_size
    return packed.reshape(1, LOGICAL_WORDS), layout


def unpack_output(output: object, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(output, dtype=np.float32)
    if values.size != LOGICAL_WORDS:
        raise ValueError("selective-scan output has the wrong size")
    chunks = values.reshape(REPLICAS, INVOCATION_WORDS)
    words = chunks.view(np.uint32)
    sequence = np.empty((layout.channels, layout.sequence_length), dtype=np.float32)
    state = np.empty((layout.channels, layout.state_size), dtype=np.float32)
    for replica, channel_slice in enumerate(layout.channel_slices):
        count = channel_slice.stop - channel_slice.start
        expected = (MAGIC, VERSION, 0, count, layout.state_size, layout.sequence_length)
        if tuple(int(value) for value in words[replica, :6]) != expected:
            raise RuntimeError(f"selective-scan invocation {replica} returned invalid metadata")
        cursor = HEADER_WORDS
        for channel in range(channel_slice.start, channel_slice.stop):
            sequence[channel] = chunks[replica, cursor : cursor + layout.sequence_length]
            cursor += layout.sequence_length
            state[channel] = chunks[replica, cursor : cursor + layout.state_size]
            cursor += layout.state_size
    return sequence, state
