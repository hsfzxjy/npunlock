"""Offline tests for the semantic binary artifact visualizer."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from npunlock.viz import Address, ArtifactGraph, Category, Field, Span, build_npunlock_graph, show  # noqa: E402
from npunlock.viz._schema import decode_record_fields, load_schema, validate_schema  # noqa: E402


def expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_model_and_serialization() -> None:
    graph = ArtifactGraph("Dense model")
    first = graph.artifact("first", data=bytes(range(64)), name="first.bin")
    second = graph.artifact("second", data=b"symbol\0target\0", name="second.bin")
    outer = graph.node(
        "outer",
        label="Outer",
        description="Container record for the nested test entities.",
        kind="record",
        spans=[Span(first, 0, 32)],
        fields={"size": Field(32, [Span(first, 4, 8)], description="Encoded record size in bytes.")},
    )
    inner = graph.node(
        "inner",
        label="Inner",
        kind="record",
        category="data",
        spans=[Span(first, 8, 16)],
        parent=outer,
        addresses={"runtime": Address("virtual", 0x1D000000)},
    )
    overlap = graph.node(
        "overlap",
        label="Overlap",
        kind="record",
        category="reference",
        spans=[Span(first, 12, 24), Span(second, 0, 6)],
    )
    logical = graph.node("logical", label="No bytes", kind="operation")
    graph.link(inner, overlap, "references")
    graph.link(logical, inner, "scheduled_as")
    graph.validate()
    assert graph.nodes["logical"].spans == ()
    assert graph.nodes["inner"].addresses["runtime"].space == "virtual"
    assert len(graph.artifacts) == 2
    assert graph.to_dict()["nodes"][1]["parent"] == "outer"
    assert graph.to_dict()["nodes"][0]["description"].startswith("Container record")
    assert graph.to_dict()["nodes"][0]["fields"]["size"]["description"].startswith("Encoded record")

    portable = graph.dumps(include_data=True)
    restored = ArtifactGraph.loads(portable)
    assert restored.dumps(include_data=True) == portable
    metadata = graph.dumps(include_data=False)
    rebound = ArtifactGraph.loads(metadata, artifact_data={"first": first.read(), "second": second.read()})
    assert rebound.artifacts["first"].sha256 == first.sha256

    expect_error(ValueError, Span, first, 4, 4)
    expect_error(ValueError, Category, "bad", "Bad", "#xyz")
    expect_error(ValueError, Field, 1, (), None, " ")
    expect_error(ValueError, graph.node, "bad-description", description="")
    invalid = ArtifactGraph("Invalid")
    artifact = invalid.artifact("tiny", data=b"abc")
    invalid.node("past-end", spans=[Span(artifact, 2, 4)])
    expect_error(ValueError, invalid.validate)
    expect_error(ValueError, graph.link, "missing", inner, "bad")
    expect_error(ValueError, graph.contain, inner, outer)

    dangling = json.loads(metadata)
    dangling["links"].append({"source": "inner", "target": "missing", "kind": "bad", "properties": {}})
    expect_error(
        ValueError,
        ArtifactGraph.from_dict,
        dangling,
        artifact_data={"first": first.read(), "second": second.read()},
    )


def test_server_ranges_and_lifecycle() -> None:
    graph = ArtifactGraph("Server")
    artifact = graph.artifact("a/b", data=bytes(range(256)) * 400, name="large.bin")
    graph.node("all", spans=[Span(artifact, 0, artifact.size)])
    opened: list[str] = []
    session = show(graph, block=False, open_browser=True, _browser_open=opened.append)
    try:
        assert opened == [session.url]
        assert session.url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(session.url + "api/graph", timeout=3) as response:
            metadata = json.load(response)
        assert metadata["format"] == "npunlock.artifact-graph"
        assert "data_base64" not in metadata["artifacts"][0]
        url = session.url + "api/artifacts/a%2Fb/bytes?offset=65530&length=20"
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.read() == artifact.read(65530, 65550)
        excessive = session.url + "api/artifacts/a%2Fb/bytes?offset=0&length=65537"
        try:
            urllib.request.urlopen(excessive, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == 416
        else:
            raise AssertionError("server accepted an excessive byte range")
        request = urllib.request.Request(session.url + "api/graph", data=b"x", method="POST")
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == 405
        else:
            raise AssertionError("server accepted a write method")
    finally:
        session.close()
    assert session.closed
    session.close()

    def failed_browser(_url: str) -> None:
        raise RuntimeError("browser launch failed")

    viewer_threads = sum(thread.name == "npunlock-viz" for thread in threading.enumerate())
    expect_error(
        RuntimeError,
        show,
        graph,
        block=False,
        open_browser=True,
        _browser_open=failed_browser,
    )
    assert sum(thread.name == "npunlock-viz" for thread in threading.enumerate()) == viewer_threads


def test_windows_ctrl_c_stops_blocking_viewer() -> None:
    if sys.platform != "win32":
        return
    code = (
        "import ctypes, sys; "
        "ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False); "
        f"sys.path.insert(0, {str(ROOT / 'python')!r}); "
        "from npunlock.viz import ArtifactGraph, show; "
        "g = ArtifactGraph('ctrl-c'); "
        "g.artifact('a', data=b'x'); "
        "print('READY', flush=True); "
        "show(g, open_browser=False)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        time.sleep(0.3)
        os.kill(process.pid, signal.CTRL_C_EVENT)
        assert process.wait(timeout=5) == 0
        assert process.stderr is not None
        assert "Traceback" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_npunlock_fixture_annotations() -> None:
    graph = build_npunlock_graph(
        ROOT / "tests/fixtures/npu3720/add1-shared-1x16.blob",
        ROOT / "tests/fixtures/npu3720/add1-fp16.elf",
    )
    graph.validate()
    assert len(graph.artifacts) == 2
    assert sum(node.kind == "ActKernelRange" for node in graph.nodes.values()) == 2
    assert sum(node.kind == "ActKernelInvocation" for node in graph.nodes.values()) == 2
    assert sum(link.kind == "selects_range" for link in graph.links) == 2
    assert sum(link.kind == "selects_code" for link in graph.links) == 2
    assert graph.nodes["native.range.0"].description is not None
    assert graph.nodes["native.range.0"].fields["code_size"].description is not None
    exact = [link for link in graph.links if link.kind == "byte_identical_to"]
    assert len(exact) == 1
    assert exact[0].properties["evidence"] == "exact executable-image byte comparison"
    assert graph.nodes[exact[0].source].spans[1].artifact.id == "kernel"
    assert graph.nodes[exact[0].target].spans[0].artifact.id == "native"


def test_bundled_semantic_schemas() -> None:
    schemas = files("npunlock.viz").joinpath("schemas")
    descriptor = json.loads(schemas.joinpath("descriptor.schema.json").read_text(encoding="utf-8"))
    assert descriptor["$id"].endswith("/viz-descriptor-v1.json")

    elf_schema = load_schema("elf.json")
    act_schema = load_schema("npu3720-act-compiler-8.3.json")
    assert validate_schema(elf_schema) is elf_schema
    assert validate_schema(act_schema) is act_schema
    expect_error(ValueError, load_schema, "outside.json")

    data = (ROOT / "tests/fixtures/npu3720/add1-fp16.elf").read_bytes()
    graph = ArtifactGraph("Schema decoding")
    artifact = graph.artifact("elf", data=data)
    fields = decode_record_fields(elf_schema, "elf32_header", artifact, data, 0)
    assert fields["class"].value == "ELF32"
    assert fields["entry"].spans[0].start == 24
    assert fields["entry"].spans[0].size == 4
    assert fields["entry"].display == hex(fields["entry"].value)

    truncated = ArtifactGraph("Truncated")
    tiny = truncated.artifact("tiny", data=b"\x00" * 8)
    expect_error(ValueError, decode_record_fields, elf_schema, "elf32_header", tiny, tiny.read(), 0)

    unknown_encoding = json.loads(json.dumps(elf_schema))
    unknown_encoding["records"]["elf32_header"]["fields"][0]["encoding"] = "python-expression"
    expect_error(ValueError, validate_schema, unknown_encoding)

    outside_record = json.loads(json.dumps(elf_schema))
    outside_record["records"]["elf32_header"]["fields"][0]["offset"] = 52
    expect_error(ValueError, validate_schema, outside_record)


def test_dense_model() -> None:
    graph = ArtifactGraph("Dense annotations")
    artifact = graph.artifact("dense", data=bytes(range(256)) * 32)
    root = graph.node("root", label="Root with a long name " + "x" * 160)
    parent = root
    for index in range(1_200):
        start = index % (artifact.size - 64)
        spans = [Span(artifact, start, start + 64)] if index < 900 else []
        node = graph.node(
            f"node-{index}",
            label=f"Entity {index} " + "long-name-" * 12,
            spans=spans,
            parent=parent if index < 40 else root,
        )
        if index < 40:
            parent = node
    graph.link(graph.nodes["node-1199"], graph.nodes["node-1"], "dense_reference")
    graph.validate()
    serialized = graph.to_dict()
    assert len(serialized["nodes"]) == 1_201
    assert len(serialized["links"]) == 1


def test_bundled_frontend() -> None:
    static = files("npunlock.viz").joinpath("static")
    for name in ("index.html", "app.css", "app.js"):
        content = static.joinpath(name).read_bytes()
        assert content
    javascript = static.joinpath("app.js").read_text(encoding="utf-8")
    assert "class Viewer" in javascript
    assert "api/artifacts/" in javascript
    assert "field-highlight" in javascript
    stylesheet = static.joinpath("app.css").read_text(encoding="utf-8")
    assert "color-scheme: light" in stylesheet
    assert "color-scheme: dark" not in stylesheet
    index = static.joinpath("index.html").read_text(encoding="utf-8")
    assert index.count('class="hex-label"') == 16
    assert "hex-head-viewport" in index
    schemas = files("npunlock.viz").joinpath("schemas")
    for name in ("descriptor.schema.json", "elf.json", "npu3720-act-compiler-8.3.json"):
        assert schemas.joinpath(name).read_bytes()


def main() -> None:
    test_model_and_serialization()
    test_server_ranges_and_lifecycle()
    test_windows_ctrl_c_stops_blocking_viewer()
    test_npunlock_fixture_annotations()
    test_bundled_semantic_schemas()
    test_dense_model()
    test_bundled_frontend()
    print("visualizer tests passed")


if __name__ == "__main__":
    main()
