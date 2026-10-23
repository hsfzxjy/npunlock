"""Offline tests for the semantic binary artifact visualizer."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from npunlock.viz import Address, ArtifactGraph, Field, Span, build_npunlock_graph, show  # noqa: E402


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
        kind="record",
        spans=[Span(first, 0, 32)],
        fields={"size": Field(32, [Span(first, 4, 8)])},
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

    portable = graph.dumps(include_data=True)
    restored = ArtifactGraph.loads(portable)
    assert restored.dumps(include_data=True) == portable
    metadata = graph.dumps(include_data=False)
    rebound = ArtifactGraph.loads(metadata, artifact_data={"first": first.read(), "second": second.read()})
    assert rebound.artifacts["first"].sha256 == first.sha256

    expect_error(ValueError, Span, first, 4, 4)
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
    exact = [link for link in graph.links if link.kind == "byte_identical_to"]
    assert len(exact) == 1
    assert exact[0].properties["evidence"] == "exact executable-image byte comparison"
    assert graph.nodes[exact[0].source].spans[1].artifact.id == "kernel"
    assert graph.nodes[exact[0].target].spans[0].artifact.id == "native"


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


def main() -> None:
    test_model_and_serialization()
    test_server_ranges_and_lifecycle()
    test_npunlock_fixture_annotations()
    test_dense_model()
    test_bundled_frontend()
    print("visualizer tests passed")


if __name__ == "__main__":
    main()
