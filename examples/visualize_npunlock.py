"""Visualize retained native-graph and SHAVE-kernel ELF fixtures."""

from pathlib import Path

from npunlock.viz import build_npunlock_graph, show


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    graph = build_npunlock_graph(
        ROOT / "tests/fixtures/npu3720/add1-shared-1x16.blob",
        ROOT / "tests/fixtures/npu3720/add1-fp16.elf",
        title="Retained add-one patch proof",
    )
    show(graph)
