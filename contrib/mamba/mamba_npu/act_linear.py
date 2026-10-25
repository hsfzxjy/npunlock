from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .packing import INVOCATION_WORDS as SCAN_INVOCATION_WORDS
from .packing import pack_dynamic, pack_model, unpack_output


MAGIC = 0x4C494E31
VERSION = 1
REPLICAS = 4
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
        activation = npu.input("linear_activation", shape=(1, LOGICAL_WORDS), dtype="f32")
        weights = npu.input("linear_weights", shape=(1, LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            activation,
            weights,
            source=KERNEL_PATH.read_bytes(),
            carrier="Subtract",
            _name="linear_output",
        )
        started = perf_counter()
        self.program = npu.compile(
            npu.Graph([activation, weights], [output], name="mamba_act_linear_f32"),
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
        )
        self.build_ms = (perf_counter() - started) * 1000.0
        self.inputs = self.program.shared_inputs()
        self.output = self.program.shared_output()

    def run(
        self,
        activation: object,
        weight: object,
        residual: object | None = None,
        *,
        rms_weight: object | None = None,
        rms_epsilon: float = 1e-5,
    ) -> tuple[np.ndarray, dict[str, float]]:
        vector = np.asarray(activation, dtype=np.float32).reshape(-1)
        matrix = np.asarray(weight, dtype=np.float32)
        residual_value = None if residual is None else np.asarray(residual, dtype=np.float32).reshape(-1)
        norm_weight = None if rms_weight is None else np.asarray(rms_weight, dtype=np.float32).reshape(-1)
        if norm_weight is not None and (norm_weight.size != vector.size or rms_epsilon <= 0.0):
            raise ValueError("RMSNorm weight must match the activation and epsilon must be positive")
        flags = (FLAG_RESIDUAL if residual_value is not None else 0) | (
            FLAG_RMS_NORM if norm_weight is not None else 0
        )
        result, _, _, timing = self._run(
            vector,
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
        vector = np.asarray(activation, dtype=np.float32).reshape(-1)
        conv_weight = np.asarray(convolution_weight, dtype=np.float32).reshape(vector.size, 4)
        conv_bias = np.asarray(convolution_bias, dtype=np.float32).reshape(-1)
        if conv_bias.size != vector.size:
            raise ValueError("convolution bias must contain one value per input channel")
        if initial_state is None:
            state = np.zeros((vector.size, 4), dtype=np.float32)
        else:
            state = np.asarray(initial_state, dtype=np.float32).reshape(vector.size, 4)
        result, convolution_output, final_state, timing = self._run(
            vector,
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
        vector: np.ndarray,
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
            or matrix.shape[0] != vector.size
            or matrix.shape[1] == 0
            or not 1 <= vector.size <= 1536
        ):
            raise ValueError("weight must have shape (1..1536 input features, output features)")
        if residual is not None and residual.size != matrix.shape[1]:
            raise ValueError("residual must contain one value per output feature")
        capacity = (INVOCATION_WORDS - HEADER_WORDS) // vector.size
        per_submission = capacity * REPLICAS
        result = np.empty(matrix.shape[1], dtype=np.float32)
        total_execution = total_upload = total_download = 0.0
        submissions = 0
        final_state = None
        convolution_output = None
        for begin in range(0, matrix.shape[1], per_submission):
            activation_pack = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
            weight_pack = np.zeros_like(activation_pack)
            awords = activation_pack.view(np.uint32)
            wwords = weight_pack.view(np.uint32)
            counts = []
            for replica in range(REPLICAS):
                replica_begin = begin + replica * capacity
                count = max(0, min(capacity, matrix.shape[1] - replica_begin))
                counts.append(count)
                awords[replica, :4] = (MAGIC, VERSION, vector.size, count)
                awords[replica, 4] = flags
                activation_pack[replica, 5] = rms_epsilon
                wwords[replica, :4] = (MAGIC, VERSION, vector.size, count)
                activation_pack[replica, HEADER_WORDS : HEADER_WORDS + vector.size] = vector
                cursor = HEADER_WORDS + vector.size
                if residual is not None and count:
                    activation_pack[replica, cursor : cursor + count] = residual[
                        replica_begin : replica_begin + count
                    ]
                    cursor += count
                if rms_weight is not None:
                    activation_pack[replica, cursor : cursor + vector.size] = rms_weight
                    cursor += vector.size
                if convolution_state is not None:
                    activation_pack[replica, cursor : cursor + vector.size * 4] = (
                        convolution_state.reshape(-1)
                    )
                    cursor += vector.size * 4
                    activation_pack[replica, cursor : cursor + vector.size * 4] = (
                        convolution_weight.reshape(-1)
                    )
                    cursor += vector.size * 4
                    activation_pack[replica, cursor : cursor + vector.size] = convolution_bias
                    cursor += vector.size
                if cursor > INVOCATION_WORDS:
                    raise ValueError("linear auxiliary data exceeds the validated invocation span")
                if count:
                    weight_pack[replica, HEADER_WORDS : HEADER_WORDS + count * vector.size] = matrix[
                        :, replica_begin : replica_begin + count
                    ].T.reshape(-1)
            started = perf_counter()
            np.copyto(self.inputs["linear_activation"], activation_pack.reshape(1, -1))
            np.copyto(self.inputs["linear_weights"], weight_pack.reshape(1, -1))
            total_upload += (perf_counter() - started) * 1000.0
            started = perf_counter()
            self.program.run(self.inputs, outputs={"linear_output": self.output})
            total_execution += (perf_counter() - started) * 1000.0
            started = perf_counter()
            chunks = np.asarray(self.output).reshape(REPLICAS, INVOCATION_WORDS)
            words = chunks.view(np.uint32)
            for replica, count in enumerate(counts):
                if tuple(words[replica, :3]) != (MAGIC, VERSION, count):
                    raise RuntimeError("ACT linear output metadata is invalid")
                if int(words[replica, 3]) != flags:
                    raise RuntimeError("ACT linear output mode is invalid")
                replica_begin = begin + replica * capacity
                result[replica_begin : replica_begin + count] = chunks[
                    replica, HEADER_WORDS : HEADER_WORDS + count
                ]
            if convolution_state is not None:
                final_state = chunks[
                    0, STATE_OUTPUT_OFFSET : STATE_OUTPUT_OFFSET + vector.size * 4
                ].copy().reshape(vector.size, 4)
                convolution_output = chunks[
                    0, CONV_OUTPUT_OFFSET : CONV_OUTPUT_OFFSET + vector.size
                ].copy().reshape(1, -1)
            total_download += (perf_counter() - started) * 1000.0
            submissions += 1
        return result.reshape(1, -1), convolution_output, final_state, {
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
        activation_pack = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
        weight_pack = np.zeros_like(activation_pack)
        activation_pack[:, :SCAN_INVOCATION_WORDS] = dynamic.reshape(REPLICAS, SCAN_INVOCATION_WORDS)
        weight_pack[:, :SCAN_INVOCATION_WORDS] = model.reshape(REPLICAS, SCAN_INVOCATION_WORDS)
        started = perf_counter()
        np.copyto(self.inputs["linear_activation"], activation_pack.reshape(1, -1))
        np.copyto(self.inputs["linear_weights"], weight_pack.reshape(1, -1))
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.inputs, outputs={"linear_output": self.output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        scan_output = np.asarray(self.output).reshape(REPLICAS, INVOCATION_WORDS)[
            :, :SCAN_INVOCATION_WORDS
        ].copy()
        sequence, final_state = unpack_output(scan_output, layout)
        download_ms = (perf_counter() - started) * 1000.0
        return sequence, final_state, {
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuActLinear":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
