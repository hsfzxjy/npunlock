from __future__ import annotations

import struct
from dataclasses import dataclass
from os import PathLike
from typing import Any, Sequence

from ._schema import decode_record_fields, load_schema, lookup, match_rule, node_metadata
from .model import Address, Artifact, ArtifactGraph, Node, Span


_ELF_SCHEMA = load_schema("elf.json")
_NPU3720_ACT_SCHEMA = load_schema("npu3720-act-compiler-8.3.json")


@dataclass(frozen=True, slots=True)
class _Section:
    index: int
    name: str
    header_offset: int
    section_type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int


@dataclass(frozen=True, slots=True)
class _Elf:
    elf_class: int
    header_size: int
    section_offset: int
    section_entry_size: int
    sections: tuple[_Section, ...]
    entry: int
    machine: int
    elf_type: int


def _bounded(data: bytes, offset: int, size: int, what: str) -> memoryview:
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError(f"{what} is outside the ELF file")
    return memoryview(data)[offset : offset + size]


def _cstring(data: bytes, offset: int, limit: int, what: str) -> str:
    if offset < 0 or offset >= limit:
        raise ValueError(f"{what} offset is outside its string table")
    end = data.find(b"\0", offset, limit)
    if end < 0:
        raise ValueError(f"{what} is not null-terminated")
    return data[offset:end].decode("utf-8", errors="replace")


def _parse_elf(data: bytes) -> _Elf:
    if len(data) < 16 or data[:4] != b"\x7fELF" or data[5] != 1:
        raise ValueError("artifact is not a little-endian ELF file")
    elf_class = data[4]
    if elf_class == 1:
        if len(data) < 52:
            raise ValueError("ELF32 header is truncated")
        elf_type, machine = struct.unpack_from("<HH", data, 16)
        entry, section_offset = struct.unpack_from("<II", data, 24)[0], struct.unpack_from("<I", data, 32)[0]
        header_size = struct.unpack_from("<H", data, 40)[0]
        section_entry_size, section_count, section_names = struct.unpack_from("<HHH", data, 46)
        expected_header, expected_section = 52, 40
        section_format = "<IIIIIIIIII"
    elif elf_class == 2:
        if len(data) < 64:
            raise ValueError("ELF64 header is truncated")
        elf_type, machine = struct.unpack_from("<HH", data, 16)
        entry = struct.unpack_from("<Q", data, 24)[0]
        section_offset = struct.unpack_from("<Q", data, 40)[0]
        header_size = struct.unpack_from("<H", data, 52)[0]
        section_entry_size, section_count, section_names = struct.unpack_from("<HHH", data, 58)
        expected_header, expected_section = 64, 64
        section_format = "<IIQQQQIIQQ"
    else:
        raise ValueError(f"unsupported ELF class: {elf_class}")
    if header_size != expected_header or section_entry_size != expected_section or section_count == 0:
        raise ValueError("ELF header or section-table shape is unsupported")
    _bounded(data, section_offset, section_count * section_entry_size, "section table")
    if section_names >= section_count:
        raise ValueError("ELF section-name table index is out of range")
    raw_sections = [
        struct.unpack_from(section_format, data, section_offset + index * section_entry_size)
        for index in range(section_count)
    ]
    name_raw = raw_sections[section_names]
    name_offset, name_size = int(name_raw[4]), int(name_raw[5])
    _bounded(data, name_offset, name_size, "section-name table")
    sections: list[_Section] = []
    for index, raw in enumerate(raw_sections):
        section_offset_value, section_size = int(raw[4]), int(raw[5])
        section_type = int(raw[1])
        if section_type != 8:
            _bounded(data, section_offset_value, section_size, f"section {index}")
        name = _cstring(data, name_offset + int(raw[0]), name_offset + name_size, f"section {index} name")
        sections.append(
            _Section(
                index,
                name,
                section_offset + index * section_entry_size,
                section_type,
                int(raw[2]),
                int(raw[3]),
                section_offset_value,
                section_size,
                int(raw[6]),
                int(raw[7]),
                int(raw[8]),
                int(raw[9]),
            )
        )
    return _Elf(elf_class, header_size, section_offset, section_entry_size, tuple(sections), entry, machine, elf_type)


def _section_category(name: str) -> str:
    category = match_rule(_NPU3720_ACT_SCHEMA, "section_categories", name, "")
    return category or match_rule(_ELF_SCHEMA, "section_categories", name, "unknown")


def _section_description(section: _Section) -> str:
    description = lookup(_NPU3720_ACT_SCHEMA, "section_descriptions", section.name)
    if description is None:
        description = lookup(_ELF_SCHEMA, "section_descriptions", section.name)
    if description is None:
        description = lookup(_ELF_SCHEMA, "section_type_descriptions", str(section.section_type))
    return description or lookup(_ELF_SCHEMA, "defaults", "section_description")


def _add_elf_nodes(graph: ArtifactGraph, artifact: Artifact, data: bytes, prefix: str) -> tuple[_Elf, dict[int, Node]]:
    elf = _parse_elf(data)
    root = graph.node(
        f"{prefix}.elf",
        label=f"{artifact.name} · ELF{elf.elf_class * 32}",
        properties={"byte_order": "little-endian", "file_size": len(data)},
        **node_metadata(_ELF_SCHEMA, "elf_image"),
    )
    graph.node(
        f"{prefix}.header",
        label="ELF header",
        spans=[Span(artifact, 0, elf.header_size)],
        fields=decode_record_fields(_ELF_SCHEMA, f"elf{elf.elf_class * 32}_header", artifact, data, 0),
        addresses={"entry": Address("virtual", elf.entry)},
        parent=root,
        **node_metadata(_ELF_SCHEMA, "elf_header"),
    )
    table = graph.node(
        f"{prefix}.sections",
        label=f"Section table · {len(elf.sections)} entries",
        spans=[Span(artifact, elf.section_offset, elf.section_offset + len(elf.sections) * elf.section_entry_size)],
        parent=root,
        **node_metadata(_ELF_SCHEMA, "section_table"),
    )
    section_nodes: dict[int, Node] = {}
    for section in elf.sections:
        spans = [Span(artifact, section.header_offset, section.header_offset + elf.section_entry_size)]
        if section.section_type != 8 and section.size:
            spans.append(Span(artifact, section.offset, section.offset + section.size))
        fields = decode_record_fields(
            _ELF_SCHEMA,
            f"elf{elf.elf_class * 32}_section_header",
            artifact,
            data,
            section.header_offset,
            computed={"index": section.index},
        )
        metadata = node_metadata(_ELF_SCHEMA, "section")
        metadata["category"] = _section_category(section.name)
        metadata["description"] = _section_description(section)
        node = graph.node(
            f"{prefix}.section.{section.index}",
            label=section.name or f"section {section.index} (null)",
            spans=spans,
            fields=fields,
            addresses={"section virtual": Address("virtual", section.address)} if section.address else {},
            parent=table,
            **metadata,
        )
        section_nodes[section.index] = node
    return elf, section_nodes


def _symbols(data: bytes, elf: _Elf) -> dict[tuple[int, int], tuple[str, int, int]]:
    result: dict[tuple[int, int], tuple[str, int, int]] = {}
    for section in elf.sections:
        if section.section_type != 2 or not section.entry_size or section.link >= len(elf.sections):
            continue
        names = elf.sections[section.link]
        if names.section_type != 3:
            continue
        for index in range(section.size // section.entry_size):
            offset = section.offset + index * section.entry_size
            if elf.elf_class == 2 and section.entry_size >= 24:
                name_offset, _, _, section_index, value, _ = struct.unpack_from("<IBBHQQ", data, offset)
            elif elf.elf_class == 1 and section.entry_size >= 16:
                name_offset, value, _, _, _, section_index = struct.unpack_from("<IIIBBH", data, offset)
            else:
                continue
            try:
                name = _cstring(data, names.offset + name_offset, names.offset + names.size, "symbol name")
            except ValueError:
                name = "<invalid>"
            result[(section.index, index)] = (name, section_index, value)
    return result


def _relocations(data: bytes, elf: _Elf, section: _Section) -> list[tuple[int, int, int, int, int]]:
    if elf.elf_class != 2 or section.section_type != 4 or section.entry_size != 24:
        return []
    result = []
    for index in range(section.size // 24):
        offset, info, addend = struct.unpack_from("<QQq", data, section.offset + index * 24)
        result.append((index, offset, info >> 32, info & 0xFFFFFFFF, addend))
    return result


def _find(elf: _Elf, name: str) -> _Section | None:
    return next((section for section in elf.sections if section.name == name), None)


def _target_contracts(target_groups: Sequence[Sequence[Any]] | None) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if target_groups is None:
        return result
    for group_index, group in enumerate(target_groups):
        for target in group:
            invocation = int(target.invocation_index)
            tensors = getattr(target, "tensors", None)
            if tensors is not None:
                contract = [
                    {
                        "role": tensor.role,
                        "index": tensor.index,
                        "dtype": tensor.dtype,
                        "element_count": tensor.element_count,
                        "span_bytes": tensor.span_bytes,
                    }
                    for tensor in tensors
                ]
            else:
                contract = {
                    "input_count": target.input_count,
                    "element_count": target.element_count,
                    "span_bytes": target.span_bytes,
                }
            result[invocation] = {"validated_group": group_index, "validated_contract": contract}
    return result


def _annotate_native_records(
    graph: ArtifactGraph,
    artifact: Artifact,
    data: bytes,
    elf: _Elf,
    section_nodes: dict[int, Node],
    *,
    prefix: str,
    target_groups: Sequence[Sequence[Any]] | None,
) -> tuple[list[Node], list[Node]]:
    ranges = _find(elf, ".text.ActKernelRanges")
    invocations = _find(elf, ".text.ActKernelInvocations")
    kernel_text = _find(elf, ".text.KernelText")
    if not ranges or not invocations or not kernel_text:
        return [], []
    if ranges.size % 0x18 or invocations.size % 0x40:
        raise ValueError("ACT range or invocation section does not use the documented record stride")
    symbols = _symbols(data, elf)
    range_relocations = next(
        (section for section in elf.sections if section.section_type == 4 and section.info == ranges.index), None
    )
    invocation_relocations = [
        section for section in elf.sections if section.section_type == 4 and section.info == invocations.index
    ]
    range_code_addends: dict[int, int] = {}
    if range_relocations:
        for _, target_offset, symbol_index, relocation_type, addend in _relocations(data, elf, range_relocations):
            symbol = symbols.get((range_relocations.link, symbol_index))
            if target_offset % 0x18 == 8 and relocation_type == 4 and symbol and symbol[0] == "sym_kernelTextSection":
                range_code_addends[target_offset // 0x18] = addend
    range_nodes: list[Node] = []
    code_nodes: dict[tuple[int, int], Node] = {}
    for index in range(ranges.size // 0x18):
        offset = ranges.offset + index * 0x18
        fields = decode_record_fields(_NPU3720_ACT_SCHEMA, "act_kernel_range", artifact, data, offset)
        virtual_address = fields["code_virtual_address"].value
        code_size = fields["code_size"].value
        node = graph.node(
            f"{prefix}.range.{index}",
            label=f"ACT range {index}",
            spans=[Span(artifact, offset, offset + 0x18)],
            fields=fields,
            addresses={"SHAVE code": Address("SHAVE virtual", virtual_address)},
            properties={"extent evidence": "execution-tested inference"},
            parent=section_nodes[ranges.index],
            **node_metadata(_NPU3720_ACT_SCHEMA, "act_range"),
        )
        range_nodes.append(node)
        addend = range_code_addends.get(index)
        if addend is not None and addend >= 0 and code_size > 0 and addend + code_size <= kernel_text.size:
            key = (addend, code_size)
            code = code_nodes.get(key)
            if code is None:
                code = graph.node(
                    f"{prefix}.code.{addend:x}.{code_size:x}",
                    label=f"KernelText +{hex(addend)} · {hex(code_size)} bytes",
                    spans=[Span(artifact, kernel_text.offset + addend, kernel_text.offset + addend + code_size)],
                    properties={"section_offset": addend, "size": code_size},
                    addresses={"SHAVE code": Address("SHAVE virtual", virtual_address)},
                    parent=section_nodes[kernel_text.index],
                    **node_metadata(_NPU3720_ACT_SCHEMA, "code_image"),
                )
                code_nodes[key] = code
            graph.link(node, code, "selects_code", properties={"evidence": "KernelText relocation + observed extent"})
    contracts = _target_contracts(target_groups)
    invocation_nodes: list[Node] = []
    for index in range(invocations.size // 0x40):
        offset = invocations.offset + index * 0x40
        fields = decode_record_fields(_NPU3720_ACT_SCHEMA, "act_kernel_invocation", artifact, data, offset)
        range_index = fields["selected_range"].value
        properties = {"range_selection_confidence": "confirmed for the supported compiler family"}
        properties.update(contracts.get(index, {}))
        node = graph.node(
            f"{prefix}.invocation.{index}",
            label=f"ACT invocation {index}",
            spans=[Span(artifact, offset, offset + 0x40)],
            fields=fields,
            properties=properties,
            parent=section_nodes[invocations.index],
            **node_metadata(_NPU3720_ACT_SCHEMA, "act_invocation"),
        )
        invocation_nodes.append(node)
        if range_index < len(range_nodes):
            graph.link(
                node, range_nodes[range_index], "selects_range", properties={"evidence": "confirmed index field"}
            )
    for relocation_section in ([range_relocations] if range_relocations else []) + invocation_relocations:
        assert relocation_section is not None
        parent = section_nodes[relocation_section.index]
        for index, target_offset, symbol_index, relocation_type, addend in _relocations(data, elf, relocation_section):
            entry_offset = relocation_section.offset + index * 24
            symbol = symbols.get((relocation_section.link, symbol_index))
            symbol_name = symbol[0] if symbol else f"symbol {symbol_index}"
            relocation = graph.node(
                f"{prefix}.relocation.{relocation_section.index}.{index}",
                label=f"{relocation_section.name}[{index}] · {symbol_name}",
                spans=[Span(artifact, entry_offset, entry_offset + 24)],
                fields=decode_record_fields(_NPU3720_ACT_SCHEMA, "elf64_rela", artifact, data, entry_offset),
                properties={"symbol": symbol_name},
                parent=parent,
                **node_metadata(_NPU3720_ACT_SCHEMA, "rela"),
            )
            if relocation_section.info == ranges.index and target_offset // 0x18 < len(range_nodes):
                graph.link(relocation, range_nodes[target_offset // 0x18], "patches_field")
            if relocation_section.info == invocations.index and target_offset // 0x40 < len(invocation_nodes):
                graph.link(relocation, invocation_nodes[target_offset // 0x40], "patches_field")
            if symbol and symbol[1] in section_nodes:
                graph.link(relocation, section_nodes[symbol[1]], "references_section")
    return range_nodes, list(code_nodes.values())


def build_npunlock_graph(
    native: bytes | bytearray | memoryview | str | PathLike[str],
    kernel: bytes | bytearray | memoryview | str | PathLike[str] | None = None,
    *,
    title: str = "npunlock artifacts",
    native_name: str = "native graph",
    kernel_name: str = "SHAVE kernel",
    target_groups: Sequence[Sequence[Any]] | None = None,
) -> ArtifactGraph:
    """Build a hardware-independent semantic view of npunlock ELF artifacts.

    ``target_groups`` may be the result of ``NativeLibraries.discover_patch_targets``
    or its v2 counterpart. It enriches invocation properties using the project's
    validated parser; the structural demo itself does not require the native DLL.
    """

    graph = ArtifactGraph(title)
    native_artifact = graph.artifact("native", data=native, name=native_name)
    native_data = native_artifact.read()
    native_elf, native_sections = _add_elf_nodes(graph, native_artifact, native_data, "native")
    _, code_nodes = _annotate_native_records(
        graph,
        native_artifact,
        native_data,
        native_elf,
        native_sections,
        prefix="native",
        target_groups=target_groups,
    )
    if kernel is not None:
        kernel_artifact = graph.artifact("kernel", data=kernel, name=kernel_name)
        kernel_data = kernel_artifact.read()
        kernel_elf, kernel_sections = _add_elf_nodes(graph, kernel_artifact, kernel_data, "kernel")
        text = _find(kernel_elf, ".text")
        if text and text.size:
            text_node = kernel_sections[text.index]
            text_bytes = kernel_data[text.offset : text.offset + text.size]
            for code in code_nodes:
                span = code.spans[0]
                if span.size == len(text_bytes) and native_artifact.read(span.start, span.end) == text_bytes:
                    graph.link(
                        text_node,
                        code,
                        "byte_identical_to",
                        properties={"evidence": "exact executable-image byte comparison"},
                    )
    graph.validate()
    return graph


__all__ = ["build_npunlock_graph"]
