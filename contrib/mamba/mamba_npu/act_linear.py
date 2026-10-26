from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .packing import INVOCATION_WORDS as SCAN_INVOCATION_WORDS
from .packing import REPLICAS as SCAN_REPLICAS
from .packing import pack_dynamic, pack_model, unpack_output


MAGIC = 0x4C494E31
PROJECTED_DELTA_MAGIC = 0x4D414432
VERSION = 1
GROUP_REPLICAS = 4
REPLICAS = 8
INVOCATION_WORDS = 65536
LOGICAL_WORDS = REPLICAS * INVOCATION_WORDS
HEADER_WORDS = 16
STATE_OUTPUT_OFFSET = 4096
CONV_OUTPUT_OFFSET = 12288
FLAG_RESIDUAL = 1
FLAG_RMS_NORM = 2
FLAG_CONVOLUTION = 4
KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "linear_f32.c"


class NpuActLinear:
    def __init__(self, *, movi_dll_dir: str | Path, native_dir: str | Path | None = None):
        npu.configure(movi_dll_dir=movi_dll_dir)
        activation = npu.input("linear_activation", shape=(1, LOGICAL_WORDS), dtype="f32")
        weights = npu.input("linear_weights", shape=(1, LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            activation,
            weights,
            source=KERNEL_PATH.read_bytes(),
            carrier="Subtract",
            _name="linear_output",
        )
        prepared = npu.prepare(
            npu.Graph([activation, weights], [output], name="mamba_act_linear_f32"),
            native_dir=native_dir,
        )
        if len(prepared.groups) != REPLICAS // GROUP_REPLICAS:
            raise RuntimeError(
                f"ACT linear carrier exposed {len(prepared.groups)} groups, "
                f"expected {REPLICAS // GROUP_REPLICAS}"
            )
        for group in prepared.groups:
            if (
                group.input_dtypes != ("f32", "f32")
                or group.output_dtype != "f32"
                or len(group.targets) != GROUP_REPLICAS
                or group.per_invocation_input_element_counts
                != (INVOCATION_WORDS, INVOCATION_WORDS)
                or group.per_invocation_output_element_count != INVOCATION_WORDS
            ):
                raise RuntimeError(
                    "ACT linear carrier partition contract changed: "
                    f"inputs={group.per_invocation_input_element_counts!r}, "
                    f"output={group.per_invocation_output_element_count!r}, "
                    f"targets={len(group.targets)}"
                )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.inputs = self.program.shared_inputs()
        self.output = self.program.shared_output()
        self._alternate_inputs = self.program.shared_inputs()
        self._alternate_output = self.program.shared_output()
        self._buffer_sets = (
            (self.inputs, self.output),
            (self._alternate_inputs, self._alternate_output),
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mamba-npu")

    def run(
        self,
        activation: object,
        weight: object,
        residual: object | None = None,
        *,
        rms_weight: object | None = None,
        rms_epsilon: float = 1e-5,
    ) -> tuple[np.ndarray, dict[str, float]]:
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or not 1 <= values.shape[0] <= 16:
            raise ValueError("activation must have shape (1..16, input_features)")
        matrix = np.asarray(weight, dtype=np.float32)
        residual_value = None if residual is None else np.asarray(residual, dtype=np.float32)
        if residual_value is not None and residual_value.ndim == 1:
            residual_value = residual_value.reshape(1, -1)
        norm_weight = None if rms_weight is None else np.asarray(rms_weight, dtype=np.float32).reshape(-1)
        if norm_weight is not None and (norm_weight.size != values.shape[1] or rms_epsilon <= 0.0):
            raise ValueError("RMSNorm weight must match the activation and epsilon must be positive")
        flags = (FLAG_RESIDUAL if residual_value is not None else 0) | (
            FLAG_RMS_NORM if norm_weight is not None else 0
        )
        result, _, _, timing = self._run(
            values,
            matrix,
            flags=flags,
            residual=residual_value,
            rms_weight=norm_weight,
            rms_epsilon=rms_epsilon,
        )
        return result, timing

    def run_convolution(
        self,
        activation: object,
        convolution_weight: object,
        convolution_bias: object,
        projection_weight: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or not 1 <= values.shape[0] <= 16:
            raise ValueError("activation must have shape (1..16, input_features)")
        input_count = values.shape[1]
        conv_weight = np.asarray(convolution_weight, dtype=np.float32).reshape(input_count, 4)
        conv_bias = np.asarray(convolution_bias, dtype=np.float32).reshape(-1)
        if conv_bias.size != input_count:
            raise ValueError("convolution bias must contain one value per input channel")
        if initial_state is None:
            state = np.zeros((input_count, 4), dtype=np.float32)
        else:
            state = np.asarray(initial_state, dtype=np.float32).reshape(input_count, 4)
        result, convolution_output, final_state, timing = self._run(
            values,
            np.asarray(projection_weight, dtype=np.float32),
            flags=FLAG_CONVOLUTION,
            convolution_state=state,
            convolution_weight=conv_weight,
            convolution_bias=conv_bias,
        )
        if convolution_output is None or final_state is None:
            raise RuntimeError("ACT linear convolution did not return its recurrent state")
        return result, convolution_output, final_state, timing

    def _run(
        self,
        values: np.ndarray,
        matrix: np.ndarray,
        *,
        flags: int,
        residual: np.ndarray | None = None,
        rms_weight: np.ndarray | None = None,
        rms_epsilon: float = 1e-5,
        convolution_state: np.ndarray | None = None,
        convolution_weight: np.ndarray | None = None,
        convolution_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, float]]:
        if (
            matrix.ndim != 2
            or matrix.shape[0] != values.shape[1]
            or matrix.shape[1] == 0
            or not 1 <= values.shape[1] <= 1536
        ):
            raise ValueError("weight must have shape (1..1536 input features, output features)")
        sequence_length, input_count = values.shape
        if residual is not None and residual.shape != (sequence_length, matrix.shape[1]):
            raise ValueError("residual must match activation rows and output features")
        weight_capacity = (INVOCATION_WORDS - HEADER_WORDS) // input_count
        fixed_activation_words = HEADER_WORDS + values.size
        if rms_weight is not None:
            fixed_activation_words += input_count
        if convolution_state is not None:
            fixed_activation_words += input_count * 9
        if convolution_state is not None:
            output_limit = STATE_OUTPUT_OFFSET
        elif rms_weight is not None:
            output_limit = 32768
        else:
            output_limit = INVOCATION_WORDS
        output_capacity = (output_limit - HEADER_WORDS) // sequence_length
        capacity = min(weight_capacity, output_capacity)
        while True:
            candidate = capacity + 1
            if candidate > output_capacity:
                break
            activation_words = fixed_activation_words
            if residual is not None:
                activation_words += candidate * sequence_length
            activation_words = (activation_words + 3) & ~3
            activation_words += max(0, candidate - weight_capacity) * input_count
            if activation_words > INVOCATION_WORDS:
                break
            capacity = candidate
        if capacity == 0:
            raise ValueError("linear output cannot fit the validated invocation span")
        active_replicas = GROUP_REPLICAS if convolution_state is not None else REPLICAS
        per_submission = capacity * active_replicas
        result = np.empty((sequence_length, matrix.shape[1]), dtype=np.float32)
        total_execution = total_upload = total_download = 0.0
        submissions = 0
        final_state = None
        convolution_output = None
        flat_values = values.reshape(-1)
        flat_convolution_state = (
            None if convolution_state is None else convolution_state.reshape(-1)
        )
        flat_convolution_weight = (
            None if convolution_weight is None else convolution_weight.reshape(-1)
        )

        def pack_submission(inputs: object, begin: int) -> tuple[list[int], float]:
            started = perf_counter()
            activation_pack = np.asarray(inputs["linear_activation"]).reshape(
                REPLICAS, INVOCATION_WORDS
            )
            weight_pack = np.asarray(inputs["linear_weights"]).reshape(
                REPLICAS, INVOCATION_WORDS
            )
            awords = activation_pack.view(np.uint32)
            wwords = weight_pack.view(np.uint32)
            awords[:, 0] = MAGIC
            awords[:, 1] = VERSION
            awords[:, 2] = input_count
            awords[:, 4] = flags
            activation_pack[:, 5] = rms_epsilon
            awords[:, 7] = sequence_length
            wwords[:, 0] = MAGIC
            wwords[:, 1] = VERSION
            wwords[:, 2] = input_count
            activation_pack[:, HEADER_WORDS : HEADER_WORDS + values.size] = flat_values
            common_auxiliary_offset = HEADER_WORDS + values.size
            if residual is None and rms_weight is not None:
                activation_pack[
                    :, common_auxiliary_offset : common_auxiliary_offset + input_count
                ] = rms_weight
            if residual is None and convolution_state is not None:
                cursor = common_auxiliary_offset
                activation_pack[:, cursor : cursor + input_count * 4] = flat_convolution_state
                cursor += input_count * 4
                activation_pack[:, cursor : cursor + input_count * 4] = flat_convolution_weight
                cursor += input_count * 4
                activation_pack[:, cursor : cursor + input_count] = convolution_bias
            counts = []
            for replica in range(REPLICAS):
                replica_begin = begin + replica * capacity
                count = (
                    max(0, min(capacity, matrix.shape[1] - replica_begin))
                    if replica < active_replicas
                    else 0
                )
                counts.append(count)
                awords[replica, 3] = count
                wwords[replica, 3] = count
                weight_rows = min(count, weight_capacity)
                awords[replica, 6] = weight_rows
                cursor = HEADER_WORDS + values.size
                if residual is not None and count:
                    residual_chunk = residual[:, replica_begin : replica_begin + count]
                    activation_pack[replica, cursor : cursor + residual_chunk.size] = (
                        residual_chunk.reshape(-1)
                    )
                    cursor += residual_chunk.size
                if rms_weight is not None:
                    if residual is not None:
                        activation_pack[replica, cursor : cursor + input_count] = rms_weight
                    cursor += input_count
                if convolution_state is not None:
                    if residual is None:
                        cursor += input_count * 9
                    else:
                        activation_pack[replica, cursor : cursor + input_count * 4] = (
                            flat_convolution_state
                        )
                        cursor += input_count * 4
                        activation_pack[replica, cursor : cursor + input_count * 4] = (
                            flat_convolution_weight
                        )
                        cursor += input_count * 4
                        activation_pack[replica, cursor : cursor + input_count] = convolution_bias
                        cursor += input_count
                if cursor > INVOCATION_WORDS:
                    raise ValueError("linear auxiliary data exceeds the validated invocation span")
                cursor = (cursor + 3) & ~3
                if weight_rows:
                    weight_pack[
                        replica, HEADER_WORDS : HEADER_WORDS + weight_rows * input_count
                    ] = matrix[:, replica_begin : replica_begin + weight_rows].T.reshape(-1)
                activation_rows = count - weight_rows
                if activation_rows:
                    activation_pack[
                        replica, cursor : cursor + activation_rows * input_count
                    ] = matrix[
                        :,
                        replica_begin + weight_rows : replica_begin + count,
                    ].T.reshape(-1)
            return counts, (perf_counter() - started) * 1000.0

        def execute_submission(inputs: object, output: object) -> float:
            started = perf_counter()
            self.program.run(inputs, outputs={"linear_output": output})
            return (perf_counter() - started) * 1000.0

        begins = list(range(0, matrix.shape[1], per_submission))
        counts, upload_ms = pack_submission(self._buffer_sets[0][0], begins[0])
        total_upload += upload_ms
        for submission_index, begin in enumerate(begins):
            buffer_index = submission_index & 1
            inputs, output = self._buffer_sets[buffer_index]
            future = self._executor.submit(execute_submission, inputs, output)
            next_counts = None
            if submission_index + 1 < len(begins):
                next_inputs = self._buffer_sets[buffer_index ^ 1][0]
                try:
                    next_counts, upload_ms = pack_submission(
                        next_inputs, begins[submission_index + 1]
                    )
                    total_upload += upload_ms
                except BaseException:
                    future.result()
                    raise
            total_execution += future.result()
            started = perf_counter()
            chunks = np.asarray(output).reshape(REPLICAS, INVOCATION_WORDS)
            words = chunks.view(np.uint32)
            if (
                not np.all(words[:, 0] == MAGIC)
                or not np.all(words[:, 1] == VERSION)
                or not np.array_equal(words[:, 2], np.asarray(counts, dtype=np.uint32))
            ):
                raise RuntimeError("ACT linear output metadata is invalid")
            if not np.all(words[:, 3] == flags):
                raise RuntimeError("ACT linear output mode is invalid")
            if not np.all(words[:, 4] == sequence_length):
                raise RuntimeError("ACT linear output sequence length is invalid")
            if (
                counts[0] > 0
                and all(count == counts[0] for count in counts[:active_replicas])
                and not any(counts[active_replicas:])
            ):
                count = counts[0]
                payload = chunks[
                    :active_replicas,
                    HEADER_WORDS : HEADER_WORDS + sequence_length * count,
                ].reshape(active_replicas, sequence_length, count)
                result[:, begin : begin + active_replicas * count] = payload.transpose(
                    1, 0, 2
                ).reshape(sequence_length, active_replicas * count)
            else:
                for replica, count in enumerate(counts):
                    replica_begin = begin + replica * capacity
                    result[:, replica_begin : replica_begin + count] = chunks[
                        replica, HEADER_WORDS : HEADER_WORDS + sequence_length * count
                    ].reshape(sequence_length, count)
            if convolution_state is not None:
                final_state = chunks[
                    0, STATE_OUTPUT_OFFSET : STATE_OUTPUT_OFFSET + input_count * 4
                ].copy().reshape(input_count, 4)
                convolution_output = chunks[
                    0, CONV_OUTPUT_OFFSET : CONV_OUTPUT_OFFSET + sequence_length * input_count
                ].copy().reshape(sequence_length, input_count)
            total_download += (perf_counter() - started) * 1000.0
            submissions += 1
            if next_counts is not None:
                counts = next_counts
        return result, convolution_output, final_state, {
            "upload_ms": total_upload,
            "execution_wall_ms": total_execution,
            "download_ms": total_download,
            "submissions": float(submissions),
        }

    def run_scan(
        self,
        a: object,
        d: object,
        delta_bias: object,
        u: object,
        delta: object,
        b: object,
        c: object,
        z: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        dynamic, layout = pack_dynamic(u, delta, b, c, z, initial_state)
        model, model_layout = pack_model(a, d, delta_bias, sequence_length=layout.sequence_length)
        if model_layout != layout:
            raise ValueError("selective-scan model and dynamic layouts differ")
        started = perf_counter()
        activation_pack = np.asarray(self.inputs["linear_activation"]).reshape(
            REPLICAS, INVOCATION_WORDS
        )
        weight_pack = np.asarray(self.inputs["linear_weights"]).reshape(REPLICAS, INVOCATION_WORDS)
        activation_pack[:, :SCAN_INVOCATION_WORDS] = 0.0
        weight_pack[:, :SCAN_INVOCATION_WORDS] = 0.0
        activation_pack[:SCAN_REPLICAS, :SCAN_INVOCATION_WORDS] = dynamic.reshape(
            SCAN_REPLICAS, SCAN_INVOCATION_WORDS
        )
        weight_pack[:SCAN_REPLICAS, :SCAN_INVOCATION_WORDS] = model.reshape(
            SCAN_REPLICAS, SCAN_INVOCATION_WORDS
        )
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.inputs, outputs={"linear_output": self.output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        scan_output = np.asarray(self.output).reshape(REPLICAS, INVOCATION_WORDS)[
            :SCAN_REPLICAS, :SCAN_INVOCATION_WORDS
        ].copy()
        sequence, final_state = unpack_output(scan_output, layout)
        download_ms = (perf_counter() - started) * 1000.0
        return sequence, final_state, {
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
        }

    def run_scan_projected_delta(
        self,
        a: object,
        d: object,
        delta_bias: object,
        delta_weight: object,
        time_step: object,
        u: object,
        b: object,
        c: object,
        z: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        a_value = np.asarray(a, dtype=np.float32)
        if (
            a_value.ndim != 2
            or not 1 <= a_value.shape[0] <= 1536
            or a_value.shape[1] == 0
            or a_value.shape[1] > 16
        ):
            raise ValueError("a must have shape (channels, 1..16 state dimensions)")
        channels, state_size = a_value.shape
        d_value = np.asarray(d, dtype=np.float32).reshape(-1)
        bias_value = np.asarray(delta_bias, dtype=np.float32).reshape(-1)
        steps = np.asarray(time_step, dtype=np.float32)
        u_value = np.asarray(u, dtype=np.float32)
        z_value = np.asarray(z, dtype=np.float32)
        b_value = np.asarray(b, dtype=np.float32)
        c_value = np.asarray(c, dtype=np.float32)
        if steps.ndim != 2 or not 1 <= steps.shape[0] <= 16 or not 1 <= steps.shape[1] <= 64:
            raise ValueError("time_step must have shape (1..16 positions, 1..64 rank)")
        sequence_length, time_rank = steps.shape
        delta_matrix = np.asarray(delta_weight, dtype=np.float32)
        if delta_matrix.shape != (time_rank, channels):
            raise ValueError("delta_weight must have shape (time rank, channels)")
        if (
            d_value.size != channels
            or bias_value.size != channels
            or u_value.shape != (channels, sequence_length)
            or z_value.shape != (channels, sequence_length)
            or b_value.shape != (sequence_length, state_size)
            or c_value.shape != (sequence_length, state_size)
        ):
            raise ValueError("projected selective-scan tensors have incompatible shapes")
        if initial_state is None:
            state = np.zeros((channels, state_size), dtype=np.float32)
        else:
            state = np.asarray(initial_state, dtype=np.float32)
            if state.shape != (channels, state_size):
                raise ValueError("initial_state must have shape (channels, state dimensions)")
        dynamic_pack = np.asarray(self.inputs["linear_activation"]).reshape(
            REPLICAS, INVOCATION_WORDS
        )
        model_pack = np.asarray(self.inputs["linear_weights"]).reshape(REPLICAS, INVOCATION_WORDS)
        dynamic_words = dynamic_pack.view(np.uint32)
        model_words = model_pack.view(np.uint32)
        base, remainder = divmod(channels, REPLICAS)
        output_stride = sequence_length + state_size
        shared_words = sequence_length * time_rank + sequence_length * state_size * 2
        dynamic_stride = sequence_length * 2 + state_size
        model_stride = time_rank + state_size + 4
        channel_slices = []
        begin = 0
        started = perf_counter()
        for replica in range(REPLICAS):
            count = base + (1 if replica < remainder else 0)
            end = begin + count
            if (
                HEADER_WORDS + shared_words + count * dynamic_stride > INVOCATION_WORDS
                or HEADER_WORDS + count * model_stride > INVOCATION_WORDS
                or HEADER_WORDS
                + count * (output_stride + (sequence_length if sequence_length > 1 else 0))
                > INVOCATION_WORDS
            ):
                raise ValueError("projected selective scan exceeds the invocation capacity")
            channel_slices.append(slice(begin, end))
            dynamic_words[replica, :7] = (
                PROJECTED_DELTA_MAGIC,
                VERSION,
                count,
                state_size,
                sequence_length,
                time_rank,
                0,
            )
            model_words[replica, :7] = dynamic_words[replica, :7]
            cursor = HEADER_WORDS
            dynamic_pack[replica, cursor : cursor + steps.size] = steps.reshape(-1)
            cursor += steps.size
            dynamic_pack[replica, cursor : cursor + b_value.size] = b_value.reshape(-1)
            cursor += b_value.size
            dynamic_pack[replica, cursor : cursor + c_value.size] = c_value.reshape(-1)
            cursor += c_value.size
            dynamic_records = dynamic_pack[
                replica, cursor : cursor + count * dynamic_stride
            ].reshape(count, dynamic_stride)
            dynamic_records[:, :sequence_length] = u_value[begin:end]
            dynamic_records[:, sequence_length : sequence_length * 2] = z_value[begin:end]
            dynamic_records[:, sequence_length * 2 :] = state[begin:end]
            model_records = model_pack[
                replica, HEADER_WORDS : HEADER_WORDS + count * model_stride
            ].reshape(count, model_stride)
            model_records[:, :time_rank] = delta_matrix[:, begin:end].T
            model_records[:, time_rank : time_rank + state_size] = a_value[begin:end]
            model_records[:, time_rank + state_size] = d_value[begin:end]
            model_records[:, time_rank + state_size + 1] = bias_value[begin:end]
            begin = end
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.inputs, outputs={"linear_output": self.output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        chunks = np.asarray(self.output).reshape(REPLICAS, INVOCATION_WORDS)
        words = chunks.view(np.uint32)
        sequence = np.empty((channels, sequence_length), dtype=np.float32)
        final_state = np.empty((channels, state_size), dtype=np.float32)
        for replica, channel_slice in enumerate(channel_slices):
            count = channel_slice.stop - channel_slice.start
            expected = (
                PROJECTED_DELTA_MAGIC,
                VERSION,
                0,
                count,
                state_size,
                sequence_length,
                time_rank,
            )
            if tuple(int(value) for value in words[replica, :7]) != expected:
                raise RuntimeError("projected selective-scan output metadata is invalid")
            records = chunks[
                replica, HEADER_WORDS : HEADER_WORDS + count * output_stride
            ].reshape(count, output_stride)
            sequence[channel_slice] = records[:, :sequence_length]
            final_state[channel_slice] = records[:, sequence_length:]
        download_ms = (perf_counter() - started) * 1000.0
        return sequence, final_state, {
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.program.close()

    def __enter__(self) -> "NpuActLinear":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
