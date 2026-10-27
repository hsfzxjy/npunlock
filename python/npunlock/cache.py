from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from . import NativeLibraries, Program, ProgramContract


_CACHE_SCHEMA = "npunlock.program-cache.v1"
_MOVITOOLS_FILES = (
    "bin/moviCompile64.dll",
    "bin/moviAsm64.dll",
    "bin/moviLLD64.dll",
    "lib/mlibm.a",
)
_NATIVE_FILES = ("npunlock.dll", "npunlock_worker.exe")


def _hash_file_stable(path: Path) -> dict[str, object]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"dependency changed while hashing: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def dependency_identity(root: str | Path, relatives: tuple[str, ...]) -> list[dict[str, object]]:
    directory = Path(root).expanduser().resolve()
    result: list[dict[str, object]] = []
    for relative in relatives:
        path = directory / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"required dependency not found: {path}")
        result.append({"name": relative.replace("\\", "/"), **_hash_file_stable(path)})
    return result


def movitools_identity(root: str | Path) -> list[dict[str, object]]:
    return dependency_identity(root, _MOVITOOLS_FILES)


def native_runtime_identity(root: str | Path) -> list[dict[str, object]]:
    return dependency_identity(root, _NATIVE_FILES)


def worker_identity(worker: str | os.PathLike[str] | None) -> dict[str, object]:
    """Identify an exact caller worker without making its local path portable state."""

    if worker is None:
        return {"source": "bundled"}
    value = os.fspath(worker)
    if not value:
        raise ValueError("worker executable path must not be empty")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"worker executable not found: {path}")
    return {"source": "caller", **_hash_file_stable(path)}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheStatus:
    state: str
    diagnostic: str | None = None
    load_ms: float = 0.0


class ProgramCache:
    """Fail-closed content-addressed cache for exported Program bundles."""

    def __init__(self, directory: str | Path):
        value = os.fspath(directory)
        if not value:
            raise ValueError("program cache directory must not be empty")
        self.directory = Path(value).expanduser().resolve()
        if self.directory.exists() and not self.directory.is_dir():
            raise ValueError(f"program cache path is not a directory: {self.directory}")
        self.status = CacheStatus("miss")

    def _metadata_path(self, request: Mapping[str, object]) -> Path:
        return self.directory / f"request-{canonical_sha256(request)}.json"

    def load(
        self,
        request: Mapping[str, object],
        *,
        libraries: NativeLibraries,
        timeout_ms: int,
        contract: ProgramContract,
    ) -> Program | None:
        metadata_path = self._metadata_path(request)
        if not metadata_path.is_file():
            self.status = CacheStatus("miss")
            return None
        started = perf_counter()
        program = None
        try:
            if metadata_path.stat().st_size > 1024 * 1024:
                raise ValueError("cache metadata exceeds 1 MiB")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or set(metadata) != {"schema", "request", "program"}:
                raise ValueError("cache metadata fields are invalid")
            if metadata["schema"] != _CACHE_SCHEMA or metadata["request"] != request:
                raise ValueError("cache request identity is stale")
            entry = metadata["program"]
            if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
                raise ValueError("cache program entry is invalid")
            digest = entry["sha256"]
            size = entry["size"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or entry["path"] != f"program-{digest}.npu"
            ):
                raise ValueError("cache program identity is invalid")
            path = self.directory / entry["path"]
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError("cached program is missing or has the wrong size")
            if _hash_file_stable(path)["sha256"] != digest:
                raise ValueError("cached program hash is invalid")
            from . import load

            program = load(path, timeout_ms=timeout_ms, libraries=libraries)
            expected_provenance = request.get("provenance")
            if program.provenance is None or program.provenance.to_json() != expected_provenance:
                raise ValueError("cached program driver/compiler provenance is stale")
            contract.validate(program)
            self.status = CacheStatus("hit", load_ms=(perf_counter() - started) * 1000.0)
            return program
        except Exception as exc:
            if program is not None:
                program.close()
            self.status = CacheStatus("invalid", str(exc), (perf_counter() - started) * 1000.0)
            return None

    def store(self, request: Mapping[str, object], program: Program) -> None:
        previous = self.status
        program_temp: Path | None = None
        metadata_temp: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".program-", suffix=".tmp", dir=self.directory)
            os.close(descriptor)
            program_temp = Path(temporary)
            program.export(program_temp)
            identity = _hash_file_stable(program_temp)
            digest = str(identity["sha256"])
            final_program = self.directory / f"program-{digest}.npu"
            os.replace(program_temp, final_program)
            program_temp = None
            metadata = {
                "schema": _CACHE_SCHEMA,
                "request": request,
                "program": {
                    "path": final_program.name,
                    "size": identity["size"],
                    "sha256": digest,
                },
            }
            descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".tmp", dir=self.directory)
            os.close(descriptor)
            metadata_temp = Path(temporary)
            metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(metadata_temp, self._metadata_path(request))
            metadata_temp = None
            state = "refreshed" if previous.state == "invalid" else "stored"
            self.status = CacheStatus(state, previous.diagnostic if state == "refreshed" else None, previous.load_ms)
        except Exception as exc:
            self.status = CacheStatus("write-error", str(exc), previous.load_ms)
        finally:
            for path in (program_temp, metadata_temp):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
