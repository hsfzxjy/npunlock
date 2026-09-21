import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent


def run_cmake(*arguments: str) -> None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "tools/msvc-run.ps1"),
        "cmake",
        *arguments,
    ]
    subprocess.check_call(command, cwd=ROOT)


class BuildPy(build_py):
    def run(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("npunlock native wheels currently require Windows x64")
        if struct.calcsize("P") != 8:
            raise RuntimeError("npunlock native wheels require 64-bit Python")
        super().run()
        configuration = os.environ.get("NPUNLOCK_CMAKE_CONFIG", "Release")
        native_build = Path(self.build_lib).resolve().parent / "npunlock-native"
        native_stage = Path(self.build_lib).resolve().parent / "npunlock-runtime"
        if native_stage.exists():
            shutil.rmtree(native_stage)
        run_cmake(
            "-S",
            str(ROOT),
            "-B",
            str(native_build),
            "-DNPUNLOCK_BUILD_TESTS=OFF",
        )
        run_cmake(
            "--build",
            str(native_build),
            "--config",
            configuration,
            "--target",
            "npunlock_native",
            "npunlock_worker",
        )
        run_cmake(
            "--install",
            str(native_build),
            "--config",
            configuration,
            "--prefix",
            str(native_stage),
            "--component",
            "PythonRuntime",
        )
        package_bin = Path(self.build_lib) / "npunlock/_bin"
        package_bin.mkdir(parents=True, exist_ok=True)
        for filename in ("npunlock.dll", "npunlock_worker.exe"):
            shutil.copy2(native_stage / "bin" / filename, package_bin / filename)


class BdistWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
