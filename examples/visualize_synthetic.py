"""Hardware-independent semantic artifact visualizer example."""

from npunlock.viz import Address, ArtifactGraph, Field, Span, show


def build_graph() -> ArtifactGraph:
    graph = ArtifactGraph("Synthetic artifact")
    image = graph.artifact("image", name="image.bin", data=bytes(range(256)) * 8)
    symbols = graph.artifact("symbols", name="symbols.bin", data=b"entry\0worker\0payload\0")

    header = graph.node(
        "header",
        label="Container header",
        description="Synthetic fixed-size header containing the table location and file signature.",
        kind="header",
        category="metadata",
        spans=[Span(image, 0, 64)],
        fields={
            "magic": Field(
                "00 01 02 03",
                [Span(image, 0, 4)],
                description="Four-byte signature identifying this synthetic container.",
            ),
            "table_offset": Field(
                0x80,
                [Span(image, 8, 12)],
                display="0x80",
                description="File-relative offset of the first record table.",
            ),
        },
    )
    table = graph.node(
        "records",
        label="Record table with a deliberately long descriptive name",
        description="A nested region containing fixed-size synthetic records.",
        kind="table",
        category="metadata",
        spans=[Span(image, 0x80, 0x140)],
        parent=header,
    )
    record = graph.node(
        "record-0",
        label="Record 0",
        kind="record",
        category="data",
        spans=[Span(image, 0x90, 0xB0)],
        fields={"code_size": Field(48, [Span(image, 0x9C, 0xA0)])},
        parent=table,
    )
    code = graph.node(
        "code",
        label="Executable image",
        description="One logical code entity represented by two noncontiguous byte spans.",
        kind="code",
        category="code",
        spans=[Span(image, 0x180, 0x1B0), Span(image, 0x1C0, 0x1D0)],
        addresses={"entry": Address("virtual", 0x1D000000)},
        properties={"note": "One logical entity may cover noncontiguous bytes."},
    )
    symbol = graph.node(
        "entry-symbol",
        label="entry",
        kind="symbol",
        category="reference",
        spans=[Span(symbols, 0, 6)],
    )
    logical = graph.node(
        "operation",
        label="Logical operation without physical bytes",
        kind="operation",
        category="metadata",
        properties={"state": "confirmed"},
    )
    graph.link(record, code, "uses_code")
    graph.link(symbol, code, "names")
    graph.link(logical, record, "scheduled_as")
    return graph


if __name__ == "__main__":
    show(build_graph())
