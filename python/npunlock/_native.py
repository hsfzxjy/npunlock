from __future__ import annotations

import ctypes
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class NativeError(RuntimeError):
    def __init__(
        self,
        stage: str,
        status: int,
        status_name: str,
        diagnostic: bytes,
        stdout_log: bytes = b"",
        stderr_log: bytes = b"",
    ):
        detail = diagnostic.decode("utf-8", errors="replace") if diagnostic else ""
        message = f"{stage} failed: {status_name} ({status})"
        if detail:
            message += f": {detail}"
        super().__init__(message)
        self.stage = stage
        self.status = status
        self.status_name = status_name
        self.diagnostic = diagnostic
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log


class _View(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("size", ctypes.c_size_t)]


_ReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t)


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
        ("release", _ReleaseFn),
        ("context", ctypes.c_void_p),
    ]


class _Diagnostic(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("json", _Buffer)]


class _ShaveOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("movi_dll_directory_utf8", _View),
        ("worker_executable_utf8", _View),
        ("target_cpu", _View),
        ("entry_symbol", _View),
        ("compiler_definitions", ctypes.POINTER(_View)),
        ("compiler_definition_count", ctypes.c_size_t),
        ("linker_script", _View),
        ("timeout_ms", ctypes.c_uint32),
    ]


class _ShaveResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("elf", _Buffer),
        ("stdout_log", _Buffer),
        ("stderr_log", _Buffer),
        ("diagnostic", _Diagnostic),
    ]


class _IrOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("driver_index", ctypes.c_uint32),
        ("device_index", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
        ("worker_executable_utf8", _View),
        ("build_flags", _View),
    ]


class _IrResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("selected_driver_index", ctypes.c_uint32),
        ("selected_device_index", ctypes.c_uint32),
        ("graph_extension_version", ctypes.c_uint32),
        ("compiler_version_major", ctypes.c_uint16),
        ("compiler_version_minor", ctypes.c_uint16),
        ("max_opset_version", ctypes.c_uint32),
        ("driver_version", ctypes.c_uint32),
        ("device_vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("elf_version_major", ctypes.c_uint32),
        ("elf_version_minor", ctypes.c_uint32),
        ("elf_version_patch", ctypes.c_uint32),
        ("runtime_version_major", ctypes.c_uint32),
        ("runtime_version_minor", ctypes.c_uint32),
        ("runtime_version_patch", ctypes.c_uint32),
        ("graph_blob", _Buffer),
        ("stdout_log", _Buffer),
        ("stderr_log", _Buffer),
        ("diagnostic", _Diagnostic),
    ]


class _PatchOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("image_alignment", ctypes.c_uint32),
        ("tail_padding", ctypes.c_uint32),
    ]


class _PatchTarget(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("invocation_index", ctypes.c_uint32),
        ("range_index", ctypes.c_uint32),
        ("expected_input_count", ctypes.c_uint32),
        ("expected_element_count", ctypes.c_uint64),
        ("expected_span_bytes", ctypes.c_uint64),
        ("required_contract_flags", ctypes.c_uint32),
    ]


class _PatchDiscoveredTarget(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("group_index", ctypes.c_uint32),
        ("target", _PatchTarget),
    ]


class _PatchDiscoveryResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("targets", ctypes.POINTER(_PatchDiscoveredTarget)),
        ("target_count", ctypes.c_size_t),
        ("group_count", ctypes.c_size_t),
        ("diagnostic", _Diagnostic),
    ]


class _PatchResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("graph_blob", _Buffer),
        ("report_json", _Buffer),
        ("diagnostic", _Diagnostic),
    ]


class _InferOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("driver_index", ctypes.c_uint32),
        ("device_index", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
    ]


class _InferInput(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("argument_index", ctypes.c_uint32),
        ("argument_name_utf8", _View),
        ("data", _View),
    ]


class _InferOutput(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("argument_index", ctypes.c_uint32),
        ("precision", ctypes.c_uint32),
        ("dims_count", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 5),
        ("argument_name_utf8", _Buffer),
        ("data", _Buffer),
    ]


class _InferResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("selected_driver_index", ctypes.c_uint32),
        ("selected_device_index", ctypes.c_uint32),
        ("driver_version", ctypes.c_uint32),
        ("device_vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("outputs", ctypes.POINTER(_InferOutput)),
        ("output_count", ctypes.c_size_t),
        ("diagnostic", _Diagnostic),
    ]


class _InferSessionResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("selected_driver_index", ctypes.c_uint32),
        ("selected_device_index", ctypes.c_uint32),
        ("driver_version", ctypes.c_uint32),
        ("device_vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("session", ctypes.c_void_p),
        ("diagnostic", _Diagnostic),
    ]


class _InferSharedBuffer(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
        ("implementation", ctypes.c_void_p),
        ("diagnostic", _Diagnostic),
    ]


class _InferSharedTensor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("argument_index", ctypes.c_uint32),
        ("argument_name_utf8", _View),
        ("buffer", ctypes.POINTER(_InferSharedBuffer)),
    ]


class _InferSessionRunResult(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("diagnostic", _Diagnostic)]


@dataclass(frozen=True, slots=True)
class PatchTarget:
    invocation_index: int
    range_index: int
    input_count: int
    element_count: int
    span_bytes: int
    contract_flags: int = 0x1F

    def __post_init__(self) -> None:
        u32 = ("invocation_index", "range_index", "input_count", "contract_flags")
        u64 = ("element_count", "span_bytes")
        for name in u32:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{name} must fit uint32")
        for name in u64:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"{name} must be a positive uint64")
        if self.input_count == 0:
            raise ValueError("input_count must be positive")


@dataclass(frozen=True, slots=True)
class IrCompileResult:
    graph_blob: bytes
    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int
    graph_extension_version: int
    compiler_version: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PatchResult:
    graph_blob: bytes
    report_json: bytes


@dataclass(frozen=True, slots=True)
class InferenceInput:
    selector: int | str
    data: bytes

    def __post_init__(self) -> None:
        if isinstance(self.selector, bool) or not isinstance(self.selector, (int, str)):
            raise TypeError("inference selector must be an argument index or name")
        if isinstance(self.selector, int) and not 0 <= self.selector < 0xFFFFFFFF:
            raise ValueError("inference argument index must fit uint32")
        if isinstance(self.selector, str) and not self.selector:
            raise ValueError("inference argument name must not be empty")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("inference input data must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    argument_index: int
    argument_name: str
    shape: tuple[int, ...]
    dtype: str
    data: bytes


@dataclass(frozen=True, slots=True)
class InferenceResult:
    outputs: tuple[InferenceOutput, ...]
    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int


class NativeSharedBuffer:
    def __init__(self, session: InferenceSession, native: _InferSharedBuffer):
        self._session = session
        self._native = native
        self._released = False

    @property
    def address(self) -> int:
        if self._released or not self._native.data:
            raise RuntimeError("shared buffer has been released")
        value = ctypes.cast(self._native.data, ctypes.c_void_p).value
        assert value is not None
        return value

    @property
    def size(self) -> int:
        if self._released:
            raise RuntimeError("shared buffer has been released")
        return self._native.size

    def release(self) -> None:
        if not self._released:
            self._session._libraries.infer.graphinfer_shared_buffer_release(ctypes.byref(self._native))
            self._released = True

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class InferenceSession:
    def __init__(
        self,
        libraries: NativeLibraries,
        graph_blob: bytes,
        *,
        timeout_ms: int,
    ):
        self._libraries = libraries
        graph_view, graph_owner = _owned_view(graph_blob)
        options = _InferOptions(
            ctypes.sizeof(_InferOptions),
            0xFFFFFFFF,
            0xFFFFFFFF,
            timeout_ms,
            _View(None, 0),
        )
        self._result = _InferSessionResult()
        self._result.struct_size = ctypes.sizeof(_InferSessionResult)
        status = libraries.infer.graphinfer_session_create(
            ctypes.byref(options), graph_view, ctypes.byref(self._result)
        )
        _ = graph_owner
        if status != 0:
            try:
                libraries._raise("graphinfer_session", status, self._result.diagnostic)
            finally:
                libraries.infer.graphinfer_session_result_release(ctypes.byref(self._result))
        self._closed = False

    def create_buffer(self, size: int) -> NativeSharedBuffer:
        if self._closed:
            raise RuntimeError("inference session is closed")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("shared buffer size must be a positive integer")
        native = _InferSharedBuffer()
        native.struct_size = ctypes.sizeof(_InferSharedBuffer)
        status = self._libraries.infer.graphinfer_shared_buffer_create(self._result.session, size, ctypes.byref(native))
        if status != 0:
            try:
                self._libraries._raise("graphinfer_shared_buffer", status, native.diagnostic)
            finally:
                self._libraries.infer.graphinfer_shared_buffer_release(ctypes.byref(native))
        return NativeSharedBuffer(self, native)

    @staticmethod
    def _binding_array(
        values: Iterable[tuple[int | str, NativeSharedBuffer]],
    ) -> tuple[object, list[object]]:
        items = tuple(values)
        native = (_InferSharedTensor * len(items))()
        owners: list[object] = []
        for index, (selector, buffer) in enumerate(items):
            if not isinstance(buffer, NativeSharedBuffer) or buffer._released:
                raise TypeError("shared tensor bindings require live NativeSharedBuffer values")
            if isinstance(selector, str):
                if not selector:
                    raise ValueError("shared tensor name must not be empty")
                argument_index = 0xFFFFFFFF
                name_view, name_owner = _owned_view(selector.encode("utf-8"))
            elif isinstance(selector, int) and not isinstance(selector, bool):
                if not 0 <= selector < 0xFFFFFFFF:
                    raise ValueError("shared tensor index must fit uint32")
                argument_index = selector
                name_view, name_owner = _owned_view(b"")
            else:
                raise TypeError("shared tensor selector must be an index or name")
            owners.extend((name_owner, buffer))
            native[index] = _InferSharedTensor(
                ctypes.sizeof(_InferSharedTensor),
                argument_index,
                name_view,
                ctypes.pointer(buffer._native),
            )
        return native, owners

    def infer(
        self,
        inputs: Iterable[tuple[int | str, NativeSharedBuffer]],
        outputs: Iterable[tuple[int | str, NativeSharedBuffer]],
    ) -> None:
        if self._closed:
            raise RuntimeError("inference session is closed")
        native_inputs, input_owners = self._binding_array(inputs)
        native_outputs, output_owners = self._binding_array(outputs)
        result = _InferSessionRunResult()
        result.struct_size = ctypes.sizeof(_InferSessionRunResult)
        status = self._libraries.infer.graphinfer_session_infer(
            self._result.session,
            native_inputs,
            len(native_inputs),
            native_outputs,
            len(native_outputs),
            ctypes.byref(result),
        )
        _ = (input_owners, output_owners)
        try:
            if status != 0:
                self._libraries._raise("graphinfer_session", status, result.diagnostic)
        finally:
            self._libraries.infer.graphinfer_session_infer_result_release(ctypes.byref(result))

    def close(self) -> None:
        if not self._closed:
            self._libraries.infer.graphinfer_session_result_release(ctypes.byref(self._result))
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _owned_view(data: bytes) -> tuple[_View, object | None]:
    if not data:
        return _View(None, 0), None
    owner = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return _View(ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8)), len(data)), owner


def _buffer_bytes(buffer: _Buffer) -> bytes:
    return ctypes.string_at(buffer.data, buffer.size) if buffer.data and buffer.size else b""


def _write_verbatim(stream: object, data: bytes) -> None:
    if not data:
        return
    binary = getattr(stream, "buffer", None)
    if binary is not None:
        binary.write(data)
        binary.flush()
    else:
        stream.write(data.decode("utf-8", errors="replace"))  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]


def _report_worker_streams(stdout_log: bytes, stderr_log: bytes, *, failed: bool) -> None:
    if failed:
        _write_verbatim(sys.stdout, stdout_log)
        _write_verbatim(sys.stderr, stderr_log)
    elif stderr_log:
        warnings.warn(stderr_log.decode("utf-8", errors="replace"), RuntimeWarning, stacklevel=3)


def _native_directory(directory: str | os.PathLike[str] | None) -> Path:
    if directory is not None:
        return Path(directory).resolve()
    configured = os.environ.get("NPUNLOCK_NATIVE_DIR")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parent / "_bin"


class NativeLibraries:
    def __init__(self, directory: str | os.PathLike[str] | None = None):
        native_dir = _native_directory(directory)
        if not native_dir.is_dir():
            raise FileNotFoundError(
                f"native runtime directory does not exist: {native_dir}; install a Windows "
                "wheel containing the bundled runtime or pass native_dir explicitly"
            )
        self._dll_cookie = os.add_dll_directory(str(native_dir)) if os.name == "nt" else None
        suffix = ".dll" if os.name == "nt" else ".so"
        prefix = "" if os.name == "nt" else "lib"
        path = native_dir / f"{prefix}npunlock{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"native runtime library not found: {path}")
        self._library = ctypes.CDLL(str(path))
        self.common = self._library
        self.shave = self._library
        self.ir = self._library
        self.patch = self._library
        self.infer = self._library
        self._bind()

    def _bind(self) -> None:
        self.common.npunlock_status_name.argtypes = [ctypes.c_int]
        self.common.npunlock_status_name.restype = ctypes.c_char_p
        self.shave.shavecc_compile.argtypes = [
            ctypes.POINTER(_ShaveOptions),
            _View,
            ctypes.POINTER(_ShaveResult),
        ]
        self.shave.shavecc_compile.restype = ctypes.c_int
        self.shave.shavecc_result_release.argtypes = [ctypes.POINTER(_ShaveResult)]
        self.ir.ir2blob_compile.argtypes = [
            ctypes.POINTER(_IrOptions),
            _View,
            _View,
            ctypes.POINTER(_IrResult),
        ]
        self.ir.ir2blob_compile.restype = ctypes.c_int
        self.ir.ir2blob_result_release.argtypes = [ctypes.POINTER(_IrResult)]
        self.patch.patchblob_patch.argtypes = [
            ctypes.POINTER(_PatchOptions),
            _View,
            _View,
            ctypes.POINTER(_PatchTarget),
            ctypes.c_size_t,
            ctypes.POINTER(_PatchResult),
        ]
        self.patch.patchblob_patch.restype = ctypes.c_int
        self.patch.patchblob_result_release.argtypes = [ctypes.POINTER(_PatchResult)]
        self.patch.patchblob_discover_targets.argtypes = [
            _View,
            ctypes.POINTER(_PatchDiscoveryResult),
        ]
        self.patch.patchblob_discover_targets.restype = ctypes.c_int
        self.patch.patchblob_discovery_result_release.argtypes = [ctypes.POINTER(_PatchDiscoveryResult)]
        self.infer.graphinfer_infer.argtypes = [
            ctypes.POINTER(_InferOptions),
            _View,
            ctypes.POINTER(_InferInput),
            ctypes.c_size_t,
            ctypes.POINTER(_InferResult),
        ]
        self.infer.graphinfer_infer.restype = ctypes.c_int
        self.infer.graphinfer_result_release.argtypes = [ctypes.POINTER(_InferResult)]
        self.infer.graphinfer_session_create.argtypes = [
            ctypes.POINTER(_InferOptions),
            _View,
            ctypes.POINTER(_InferSessionResult),
        ]
        self.infer.graphinfer_session_create.restype = ctypes.c_int
        self.infer.graphinfer_session_result_release.argtypes = [ctypes.POINTER(_InferSessionResult)]
        self.infer.graphinfer_shared_buffer_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_InferSharedBuffer),
        ]
        self.infer.graphinfer_shared_buffer_create.restype = ctypes.c_int
        self.infer.graphinfer_shared_buffer_release.argtypes = [ctypes.POINTER(_InferSharedBuffer)]
        self.infer.graphinfer_session_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_InferSharedTensor),
            ctypes.c_size_t,
            ctypes.POINTER(_InferSharedTensor),
            ctypes.c_size_t,
            ctypes.POINTER(_InferSessionRunResult),
        ]
        self.infer.graphinfer_session_infer.restype = ctypes.c_int
        self.infer.graphinfer_session_infer_result_release.argtypes = [ctypes.POINTER(_InferSessionRunResult)]

    def _raise(
        self,
        stage: str,
        status: int,
        diagnostic: _Diagnostic,
        stdout_log: _Buffer | None = None,
        stderr_log: _Buffer | None = None,
    ) -> None:
        raw_name = self.common.npunlock_status_name(status)
        name = raw_name.decode("ascii", errors="replace") if raw_name else "unknown"
        stdout_bytes = _buffer_bytes(stdout_log) if stdout_log is not None else b""
        stderr_bytes = _buffer_bytes(stderr_log) if stderr_log is not None else b""
        _report_worker_streams(stdout_bytes, stderr_bytes, failed=True)
        raise NativeError(
            stage,
            status,
            name,
            _buffer_bytes(diagnostic.json),
            stdout_bytes,
            stderr_bytes,
        )

    def compile_ir(
        self,
        xml: bytes,
        weights: bytes = b"",
        *,
        build_flags: str = "",
        timeout_ms: int = 20_000,
        worker: str | None = None,
    ) -> IrCompileResult:
        xml_view, xml_owner = _owned_view(xml)
        weights_view, weights_owner = _owned_view(weights)
        flags_view, flags_owner = _owned_view(build_flags.encode("utf-8"))
        worker_view, worker_owner = _owned_view(worker.encode("utf-8") if worker else b"")
        _ = (xml_owner, weights_owner, flags_owner, worker_owner)
        options = _IrOptions(
            ctypes.sizeof(_IrOptions),
            0xFFFFFFFF,
            0xFFFFFFFF,
            timeout_ms,
            worker_view,
            flags_view,
        )
        result = _IrResult()
        result.struct_size = ctypes.sizeof(_IrResult)
        status = self.ir.ir2blob_compile(ctypes.byref(options), xml_view, weights_view, ctypes.byref(result))
        try:
            if status != 0:
                self._raise(
                    "ir2blob",
                    status,
                    result.diagnostic,
                    result.stdout_log,
                    result.stderr_log,
                )
            _report_worker_streams(
                _buffer_bytes(result.stdout_log),
                _buffer_bytes(result.stderr_log),
                failed=False,
            )
            return IrCompileResult(
                _buffer_bytes(result.graph_blob),
                result.selected_driver_index,
                result.selected_device_index,
                result.driver_version,
                result.device_vendor_id,
                result.device_id,
                result.graph_extension_version,
                (result.compiler_version_major, result.compiler_version_minor),
            )
        finally:
            self.ir.ir2blob_result_release(ctypes.byref(result))

    def compile_shave(
        self,
        source: bytes,
        *,
        movi_dll_dir: str | os.PathLike[str],
        linker_script: bytes | None = None,
        definitions: Iterable[str] = (),
        timeout_ms: int = 20_000,
        worker: str | None = None,
    ) -> bytes:
        source_view, source_owner = _owned_view(source)
        directory_view, directory_owner = _owned_view(os.fspath(movi_dll_dir).encode("utf-8"))
        script_view, script_owner = _owned_view(linker_script or b"")
        worker_view, worker_owner = _owned_view(worker.encode("utf-8") if worker else b"")
        cpu_view, cpu_owner = _owned_view(b"3720xx")
        entry_view, entry_owner = _owned_view(b"controlled_act")
        definition_pairs = [_owned_view(value.encode("utf-8")) for value in definitions]
        definition_array = (_View * len(definition_pairs))(*(pair[0] for pair in definition_pairs))
        _ = (
            source_owner,
            directory_owner,
            script_owner,
            worker_owner,
            cpu_owner,
            entry_owner,
            definition_pairs,
        )
        options = _ShaveOptions(
            ctypes.sizeof(_ShaveOptions),
            directory_view,
            worker_view,
            cpu_view,
            entry_view,
            definition_array if definition_pairs else None,
            len(definition_pairs),
            script_view,
            timeout_ms,
        )
        result = _ShaveResult()
        result.struct_size = ctypes.sizeof(_ShaveResult)
        status = self.shave.shavecc_compile(ctypes.byref(options), source_view, ctypes.byref(result))
        try:
            if status != 0:
                self._raise(
                    "shavecc",
                    status,
                    result.diagnostic,
                    result.stdout_log,
                    result.stderr_log,
                )
            _report_worker_streams(
                _buffer_bytes(result.stdout_log),
                _buffer_bytes(result.stderr_log),
                failed=False,
            )
            return _buffer_bytes(result.elf)
        finally:
            self.shave.shavecc_result_release(ctypes.byref(result))

    def patch_graph(
        self,
        graph_blob: bytes,
        shave_elf: bytes,
        targets: Iterable[PatchTarget],
        *,
        image_alignment: int = 0x400,
        tail_padding: int = 0x80,
    ) -> PatchResult:
        target_values = tuple(targets)
        if not target_values:
            raise ValueError("at least one explicit patch target is required")
        native_targets = (_PatchTarget * len(target_values))(
            *(
                _PatchTarget(
                    ctypes.sizeof(_PatchTarget),
                    target.invocation_index,
                    target.range_index,
                    target.input_count,
                    target.element_count,
                    target.span_bytes,
                    target.contract_flags,
                )
                for target in target_values
            )
        )
        graph_view, graph_owner = _owned_view(graph_blob)
        elf_view, elf_owner = _owned_view(shave_elf)
        _ = (graph_owner, elf_owner)
        options = _PatchOptions(ctypes.sizeof(_PatchOptions), image_alignment, tail_padding)
        result = _PatchResult()
        result.struct_size = ctypes.sizeof(_PatchResult)
        status = self.patch.patchblob_patch(
            ctypes.byref(options),
            graph_view,
            elf_view,
            native_targets,
            len(target_values),
            ctypes.byref(result),
        )
        try:
            if status != 0:
                self._raise("patchblob", status, result.diagnostic)
            return PatchResult(_buffer_bytes(result.graph_blob), _buffer_bytes(result.report_json))
        finally:
            self.patch.patchblob_result_release(ctypes.byref(result))

    def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[PatchTarget, ...], ...]:
        graph_view, graph_owner = _owned_view(graph_blob)
        _ = graph_owner
        result = _PatchDiscoveryResult()
        result.struct_size = ctypes.sizeof(_PatchDiscoveryResult)
        status = self.patch.patchblob_discover_targets(graph_view, ctypes.byref(result))
        try:
            if status != 0:
                self._raise("patchblob discovery", status, result.diagnostic)
            groups: list[list[PatchTarget]] = [[] for _ in range(result.group_count)]
            for index in range(result.target_count):
                discovered = result.targets[index]
                if discovered.group_index >= result.group_count:
                    raise RuntimeError("patchblob returned an invalid discovered group index")
                target = discovered.target
                groups[discovered.group_index].append(
                    PatchTarget(
                        target.invocation_index,
                        target.range_index,
                        target.expected_input_count,
                        target.expected_element_count,
                        target.expected_span_bytes,
                        target.required_contract_flags,
                    )
                )
            if any(not group for group in groups):
                raise RuntimeError("patchblob returned an empty discovered target group")
            return tuple(tuple(group) for group in groups)
        finally:
            self.patch.patchblob_discovery_result_release(ctypes.byref(result))

    def infer_graph(
        self,
        graph_blob: bytes,
        inputs: Iterable[InferenceInput],
        *,
        timeout_ms: int = 20_000,
    ) -> InferenceResult:
        input_values = tuple(inputs)
        if not input_values:
            raise ValueError("at least one inference input is required")
        graph_view, graph_owner = _owned_view(graph_blob)
        native_inputs = (_InferInput * len(input_values))()
        owners: list[object] = [graph_owner]
        for index, value in enumerate(input_values):
            if not isinstance(value, InferenceInput):
                raise TypeError("inputs must contain InferenceInput values")
            if isinstance(value.selector, str):
                argument_index = 0xFFFFFFFF
                name_view, name_owner = _owned_view(value.selector.encode("utf-8"))
            else:
                argument_index = value.selector
                name_view, name_owner = _owned_view(b"")
            data_view, data_owner = _owned_view(value.data)
            owners.extend((name_owner, data_owner))
            native_inputs[index] = _InferInput(ctypes.sizeof(_InferInput), argument_index, name_view, data_view)
        _ = owners
        options = _InferOptions(
            ctypes.sizeof(_InferOptions),
            0xFFFFFFFF,
            0xFFFFFFFF,
            timeout_ms,
        )
        result = _InferResult()
        result.struct_size = ctypes.sizeof(_InferResult)
        status = self.infer.graphinfer_infer(
            ctypes.byref(options),
            graph_view,
            native_inputs,
            len(input_values),
            ctypes.byref(result),
        )
        try:
            if status != 0:
                self._raise("graphinfer", status, result.diagnostic)
            outputs = tuple(
                InferenceOutput(
                    output.argument_index,
                    _buffer_bytes(output.argument_name_utf8).decode("utf-8", errors="strict"),
                    tuple(output.dims[: output.dims_count]),
                    {1: "f32", 2: "f16"}.get(output.precision, f"precision-{output.precision}"),
                    _buffer_bytes(output.data),
                )
                for output in result.outputs[: result.output_count]
            )
            return InferenceResult(
                outputs,
                result.selected_driver_index,
                result.selected_device_index,
                result.driver_version,
                result.device_vendor_id,
                result.device_id,
            )
        finally:
            self.infer.graphinfer_result_release(ctypes.byref(result))

    def create_inference_session(self, graph_blob: bytes, *, timeout_ms: int = 20_000) -> InferenceSession:
        return InferenceSession(self, graph_blob, timeout_ms=timeout_ms)
