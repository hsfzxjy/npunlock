from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu


REPLICAS = 4
HEADER_WORDS = 16
RMS_MAGIC = 0x524D5331
RMS_INVOCATION_WORDS = 2048
RMS_LOGICAL_WORDS = REPLICAS * RMS_INVOCATION_WORDS
CONV_MAGIC = 0x434E5631
CONV_INVOCATION_WORDS = 2048
CONV_LOGICAL_WORDS = REPLICAS * CONV_INVOCATION_WORDS
CONV_WIDTH = 4
KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels"


def _carrier_contract(invocation_words: int) -> npu.CarrierContract:
    return npu.CarrierContract(
        (
            npu.ActGroupContract(
                ("f32", "f32"),
                "f32",
                (invocation_words, invocation_words),
                invocation_words,
                REPLICAS,
            ),
        ),
        mapping_indices=((0,),),
    )


class NpuActRmsNorm:
    """FP32 Mamba RMSNorm, with the full reduction replicated on each ACT invocation."""

    def __init__(self, *, movi_dll_dir: str | Path, native_dir: str | Path | None = None):
        npu.configure(movi_dll_dir=movi_dll_dir)
        values = npu.input("rms_values", shape=(1, RMS_LOGICAL_WORDS), dtype="f32")
        weights = npu.input("rms_weights", shape=(1, RMS_LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            values,
            weights,
            source=(KERNEL_DIR / "rms_norm_f32.c").read_bytes(),
            carrier="Subtract",
            _name="rms_output",
        )
        prepared = npu.prepare(
            npu.Graph([values, weights], [output], name="mamba_rms_norm_f32"),
            native_dir=native_dir,
        )
        prepared.require(_carrier_contract(RMS_INVOCATION_WORDS))
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.inputs = self.program.shared_inputs()
        self.output = self.program.shared_output()

    def run(self, values: object, weight: object, epsilon: float) -> tuple[np.ndarray, dict[str, float]]:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        scale = np.asarray(weight, dtype=np.float32).reshape(-1)
        if vector.size != scale.size or not 1 <= vector.size <= 1536:
            raise ValueError("values and weight must contain the same 1..1536 features")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        values_pack = np.zeros((REPLICAS, RMS_INVOCATION_WORDS), dtype=np.float32)
        weights_pack = np.zeros_like(values_pack)
        values_words = values_pack.view(np.uint32)
        weights_words = weights_pack.view(np.uint32)
        for replica in range(REPLICAS):
            values_words[replica, :3] = (RMS_MAGIC, 1, vector.size)
            weights_words[replica, :3] = (RMS_MAGIC, 1, vector.size)
            values_pack[replica, 3] = epsilon
            values_pack[replica, HEADER_WORDS : HEADER_WORDS + vector.size] = vector
            weights_pack[replica, HEADER_WORDS : HEADER_WORDS + vector.size] = scale
        started = perf_counter()
        np.copyto(self.inputs["rms_values"], values_pack.reshape(1, -1))
        np.copyto(self.inputs["rms_weights"], weights_pack.reshape(1, -1))
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.inputs, outputs={"rms_output": self.output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        chunks = np.asarray(self.output).reshape(REPLICAS, RMS_INVOCATION_WORDS)
        words = chunks.view(np.uint32)
        for replica in range(REPLICAS):
            if tuple(words[replica, :3]) != (RMS_MAGIC, 1, vector.size):
                raise RuntimeError("ACT RMSNorm output metadata is invalid")
        result = chunks[0, HEADER_WORDS : HEADER_WORDS + vector.size].copy()
        download_ms = (perf_counter() - started) * 1000.0
        return result.reshape(1, -1), {
            "upload_ms": upload_ms,
            "execution_wall_ms": execution_wall_ms,
            "download_ms": download_ms,
        }

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuActRmsNorm":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class NpuActCausalConv1d:
    """One-token depthwise width-4 convolution, SiLU, and explicit cache update."""

    def __init__(self, *, movi_dll_dir: str | Path, native_dir: str | Path | None = None):
        npu.configure(movi_dll_dir=movi_dll_dir)
        dynamic = npu.input("conv_dynamic", shape=(1, CONV_LOGICAL_WORDS), dtype="f32")
        model = npu.input("conv_model", shape=(1, CONV_LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            dynamic,
            model,
            source=(KERNEL_DIR / "causal_conv1d_f32.c").read_bytes(),
            carrier="Subtract",
            _name="conv_output",
        )
        prepared = npu.prepare(
            npu.Graph([dynamic, model], [output], name="mamba_causal_conv1d_f32"),
            native_dir=native_dir,
        )
        prepared.require(_carrier_contract(CONV_INVOCATION_WORDS))
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.inputs = self.program.shared_inputs()
        self.output = self.program.shared_output()

    def run(
        self,
        values: object,
        weight: object,
        bias: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        kernels = np.asarray(weight, dtype=np.float32).reshape(vector.size, CONV_WIDTH)
        biases = np.asarray(bias, dtype=np.float32).reshape(-1)
        if not 1 <= vector.size <= 1536 or biases.size != vector.size:
            raise ValueError("convolution expects 1..1536 channels and one bias per channel")
        if initial_state is None:
            state = np.zeros((vector.size, CONV_WIDTH), dtype=np.float32)
        else:
            state = np.asarray(initial_state, dtype=np.float32).reshape(vector.size, CONV_WIDTH)
        dynamic_pack = np.zeros((REPLICAS, CONV_INVOCATION_WORDS), dtype=np.float32)
        model_pack = np.zeros_like(dynamic_pack)
        dynamic_words = dynamic_pack.view(np.uint32)
        model_words = model_pack.view(np.uint32)
        bounds = [vector.size * replica // REPLICAS for replica in range(REPLICAS + 1)]
        for replica, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            count = end - begin
            dynamic_words[replica, :3] = (CONV_MAGIC, 1, count)
            model_words[replica, :3] = (CONV_MAGIC, 1, count)
            dynamic_pack[replica, HEADER_WORDS : HEADER_WORDS + count] = vector[begin:end]
            state_begin = HEADER_WORDS + count
            dynamic_pack[replica, state_begin : state_begin + count * CONV_WIDTH] = state[begin:end].reshape(-1)
            model_pack[replica, HEADER_WORDS : HEADER_WORDS + count * CONV_WIDTH] = kernels[begin:end].reshape(-1)
            bias_begin = HEADER_WORDS + count * CONV_WIDTH
            model_pack[replica, bias_begin : bias_begin + count] = biases[begin:end]
        started = perf_counter()
        np.copyto(self.inputs["conv_dynamic"], dynamic_pack.reshape(1, -1))
        np.copyto(self.inputs["conv_model"], model_pack.reshape(1, -1))
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.inputs, outputs={"conv_output": self.output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        chunks = np.asarray(self.output).reshape(REPLICAS, CONV_INVOCATION_WORDS)
        words = chunks.view(np.uint32)
        result = np.empty(vector.size, dtype=np.float32)
        final_state = np.empty((vector.size, CONV_WIDTH), dtype=np.float32)
        for replica, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            count = end - begin
            if tuple(words[replica, :3]) != (CONV_MAGIC, 1, count):
                raise RuntimeError("ACT convolution output metadata is invalid")
            result[begin:end] = chunks[replica, HEADER_WORDS : HEADER_WORDS + count]
            state_begin = HEADER_WORDS + count
            final_state[begin:end] = chunks[replica, state_begin : state_begin + count * CONV_WIDTH].reshape(
                count, CONV_WIDTH
            )
        download_ms = (perf_counter() - started) * 1000.0
        return (
            result.reshape(1, -1),
            final_state,
            {
                "upload_ms": upload_ms,
                "execution_wall_ms": execution_wall_ms,
                "download_ms": download_ms,
            },
        )

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuActCausalConv1d":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
