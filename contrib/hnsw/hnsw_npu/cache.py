from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import npunlock as npu


_CACHE_SCHEMA = "npunlock-hnsw-program-cache-v1"
_MOVITOOLS_DEPENDENCIES = (
    "bin/moviCompile64.dll",
    "bin/moviAsm64.dll",
    "bin/moviLLD64.dll",
    "lib/mlibm.a",
)
_NATIVE_RUNTIME_DEPENDENCIES = ("npunlock.dll", "npunlock_worker.exe")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dependency_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in _MOVITOOLS_DEPENDENCIES:
        path = root / Path(relative)
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError(f"MoviTools dependency changed while hashing: {path}")
        records.append(
            {
                "path": relative,
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
            }
        )
    return records


def _native_runtime_identity(native_dir: str | Path | None) -> dict[str, object]:
    if native_dir is not None:
        root = Path(native_dir).expanduser().resolve()
    elif configured := os.environ.get("NPUNLOCK_NATIVE_DIR"):
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(npu.__file__).resolve().parent / "_bin"
    files: list[dict[str, object]] = []
    for relative in _NATIVE_RUNTIME_DEPENDENCIES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"native runtime dependency not found: {path}")
        stat = path.stat()
        files.append(
            {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    return {"root": str(root), "files": files}


def provenance_identity(value: npu.IrCompileResult) -> dict[str, object]:
    return {
        "driver_index": value.driver_index,
        "device_index": value.device_index,
        "driver_version": value.driver_version,
        "vendor_id": value.vendor_id,
        "device_id": value.device_id,
        "graph_extension_version": value.graph_extension_version,
        "compiler_version": list(value.compiler_version),
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ProgramCache:
    """Content-addressed HNSW program cache bound to toolchain and NPU provenance."""

    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        source: bytes,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None,
        timeout_ms: int,
    ):
        self.status = "disabled"
        self.diagnostic: str | None = None
        self.load_ms = 0.0
        self.cache_dir: Path | None = None
        self.movi_root: Path | None = None
        self.libraries: npu.NativeLibraries | None = None
        self.provenance: dict[str, object] | None = None
        self.base: dict[str, object] | None = None
        self.metadata_path: Path | None = None
        if cache_dir is None:
            return
        cache_value = os.fspath(cache_dir)
        if not cache_value:
            raise ValueError("cache_dir must not be empty")
        self.cache_dir = Path(cache_value).expanduser().resolve()
        if self.cache_dir.exists() and not self.cache_dir.is_dir():
            raise ValueError(f"cache_dir is not a directory: {self.cache_dir}")
        self.movi_root = Path(movi_dll_dir).expanduser().resolve()
        for relative in _MOVITOOLS_DEPENDENCIES:
            dependency = self.movi_root / Path(relative)
            if not dependency.is_file():
                raise FileNotFoundError(f"MoviTools dependency not found: {dependency}")
        self.libraries = npu.NativeLibraries(native_dir)
        current = self.libraries.query_ir_provenance(timeout_ms=timeout_ms)
        self.provenance = provenance_identity(current)
        self.base = {
            "schema": _CACHE_SCHEMA,
            "kind": "hnsw-l2-search",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "movi_root": str(self.movi_root),
            "native_runtime": _native_runtime_identity(native_dir),
            "provenance": self.provenance,
        }
        self.metadata_path = self.cache_dir / f"hnsw-{_json_sha256(self.base)}.json"
        self.status = "miss"

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def _dependencies_match(self, values: object) -> bool:
        if not isinstance(values, list) or len(values) != len(_MOVITOOLS_DEPENDENCIES):
            return False
        assert self.movi_root is not None
        for relative, value in zip(_MOVITOOLS_DEPENDENCIES, values):
            if not isinstance(value, dict) or set(value) != {"path", "size", "mtime_ns", "sha256"}:
                return False
            if value["path"] != relative:
                return False
            size = value["size"]
            mtime_ns = value["mtime_ns"]
            digest = value["sha256"]
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or isinstance(mtime_ns, bool)
                or not isinstance(mtime_ns, int)
                or mtime_ns < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                return False
            stat = (self.movi_root / Path(relative)).stat()
            if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
                return False
        return True

    def load(self) -> npu.Program | None:
        if not self.enabled:
            return None
        assert self.cache_dir is not None
        assert self.metadata_path is not None
        assert self.base is not None
        assert self.libraries is not None
        assert self.provenance is not None
        if not self.metadata_path.is_file():
            return None
        started = perf_counter()
        program: npu.Program | None = None
        try:
            if self.metadata_path.stat().st_size > 1024 * 1024:
                raise ValueError("cache metadata exceeds 1 MiB")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or set(metadata) != {
                "schema",
                "base",
                "dependencies",
                "full_key",
                "program",
            }:
                raise ValueError("cache metadata fields are invalid")
            if metadata["schema"] != _CACHE_SCHEMA or metadata["base"] != self.base:
                raise ValueError("cache metadata identity is stale")
            dependencies = metadata["dependencies"]
            if not self._dependencies_match(dependencies):
                raise ValueError("cache dependency identity is stale")
            full_key = _json_sha256({"base": self.base, "dependencies": dependencies})
            if metadata["full_key"] != full_key:
                raise ValueError("cache full key is invalid")
            program_entry = metadata["program"]
            if not isinstance(program_entry, dict) or set(program_entry) != {"path", "size", "sha256"}:
                raise ValueError("cache program entry is invalid")
            program_hash = program_entry["sha256"]
            if (
                not isinstance(program_hash, str)
                or len(program_hash) != 64
                or any(character not in "0123456789abcdef" for character in program_hash)
            ):
                raise ValueError("cache program SHA-256 is invalid")
            expected_name = f"program-{program_hash}.npu"
            if program_entry["path"] != expected_name:
                raise ValueError("cache program path is invalid")
            program_path = self.cache_dir / expected_name
            size = program_entry["size"]
            if (
                not program_path.is_file()
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or program_path.stat().st_size != size
                or _sha256_file(program_path) != program_hash
            ):
                raise ValueError("cached program size or SHA-256 is invalid")
            program = npu.load(program_path, libraries=self.libraries)
            if program.artifact_manifest is None:
                raise ValueError("cached program has no artifact manifest")
            manifest = json.loads(program.artifact_manifest.decode("utf-8"))
            if manifest.get("provenance") != self.provenance:
                raise ValueError("cached program driver/compiler provenance is stale")
            self.status = "hit"
            self.diagnostic = None
            return program
        except Exception as error:
            if program is not None:
                program.close()
            self.status = "invalid"
            self.diagnostic = str(error)
            return None
        finally:
            self.load_ms = (perf_counter() - started) * 1000.0

    def start_dependency_hashes(
        self,
    ) -> tuple[ThreadPoolExecutor, Future[list[dict[str, object]]]] | None:
        if not self.enabled:
            return None
        assert self.movi_root is not None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npunlock-hnsw-cache")
        return executor, executor.submit(_dependency_records, self.movi_root)

    def store(
        self,
        program: npu.Program,
        hash_task: tuple[ThreadPoolExecutor, Future[list[dict[str, object]]]] | None,
    ) -> None:
        if not self.enabled:
            return
        assert hash_task is not None
        assert self.cache_dir is not None
        assert self.metadata_path is not None
        assert self.base is not None
        assert self.provenance is not None
        executor, future = hash_task
        program_temp: Path | None = None
        metadata_temp: Path | None = None
        previous_status = self.status
        previous_diagnostic = self.diagnostic
        try:
            dependencies = future.result()
            if program.ir_provenance is None or provenance_identity(program.ir_provenance) != self.provenance:
                raise RuntimeError("built program provenance changed during cache construction")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix=".program-", suffix=".tmp", dir=self.cache_dir)
            os.close(descriptor)
            program_temp = Path(temp_name)
            program.export(program_temp)
            program_size = program_temp.stat().st_size
            program_hash = _sha256_file(program_temp)
            program_name = f"program-{program_hash}.npu"
            program_path = self.cache_dir / program_name
            os.replace(program_temp, program_path)
            program_temp = None
            full_key = _json_sha256({"base": self.base, "dependencies": dependencies})
            metadata = {
                "schema": _CACHE_SCHEMA,
                "base": self.base,
                "dependencies": dependencies,
                "full_key": full_key,
                "program": {"path": program_name, "size": program_size, "sha256": program_hash},
            }
            descriptor, temp_name = tempfile.mkstemp(prefix=".metadata-", suffix=".tmp", dir=self.cache_dir)
            os.close(descriptor)
            metadata_temp = Path(temp_name)
            metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(metadata_temp, self.metadata_path)
            metadata_temp = None
            self.status = "refreshed" if previous_status == "invalid" else "stored"
            self.diagnostic = previous_diagnostic if self.status == "refreshed" else None
        except Exception as error:
            self.status = "write-error"
            self.diagnostic = str(error)
        finally:
            executor.shutdown(wait=True)
            for temporary in (program_temp, metadata_temp):
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
