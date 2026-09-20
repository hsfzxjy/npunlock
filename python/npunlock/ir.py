from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .graph import Graph, Node
from .tensor import Tensor, TensorSpec

_PRECISIONS = {
    "boolean": "BOOL",
    "f16": "FP16",
    "f32": "FP32",
    "f64": "FP64",
    "i8": "I8",
    "i16": "I16",
    "i32": "I32",
    "i64": "I64",
    "u8": "U8",
    "u16": "U16",
    "u32": "U32",
    "u64": "U64",
}


@dataclass(frozen=True, slots=True)
class SerializedIR:
    xml: bytes
    weights: bytes
    custom_nodes: tuple[Node, ...]


def _shape_text(spec: TensorSpec) -> str:
    return ",".join(str(dim) for dim in spec.shape)


def _attribute_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, (tuple, list)) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return ",".join(_attribute_text(item) for item in value)
    raise TypeError(f"IR attribute value is not serializable: {value!r}")


def _add_port(parent: ET.Element, port_id: int, tensor: Tensor, *, names: str | None = None) -> None:
    attributes = {"id": str(port_id), "precision": _PRECISIONS[tensor.dtype]}
    if names:
        attributes["names"] = names
    port = ET.SubElement(parent, "port", attributes)
    for dimension in tensor.shape:
        ET.SubElement(port, "dim").text = str(dimension)


def serialize_ir(graph: Graph) -> SerializedIR:
    nodes = graph.nodes
    ids = {node: index for index, node in enumerate(nodes)}
    root = ET.Element("net", {"name": graph.name, "version": "11"})
    layers = ET.SubElement(root, "layers")
    weights = bytearray()
    custom_nodes: list[Node] = []

    for node in nodes:
        node_id = ids[node]
        name = node.name or f"{node.op}_{node_id}"
        layer_type = node.op
        if node.op == "Custom":
            carrier = node.metadata.get("_carrier")
            if not isinstance(carrier, str):
                raise ValueError("custom node is missing a carrier operator")
            layer_type = carrier
            custom_nodes.append(node)
        layer = ET.SubElement(
            layers,
            "layer",
            {"id": str(node_id), "name": name, "type": layer_type, "version": "opset1"},
        )
        if node.op == "Parameter":
            spec = node.output_specs[0]
            ET.SubElement(layer, "data", {"shape": _shape_text(spec), "element_type": spec.dtype})
        elif node.op == "Const":
            array = node.metadata.get("value")
            if array is None:
                raise ValueError("constant node has no value")
            raw = array.tobytes(order="C")  # type: ignore[union-attr]
            offset = len(weights)
            weights.extend(raw)
            spec = node.output_specs[0]
            ET.SubElement(
                layer,
                "data",
                {
                    "element_type": spec.dtype,
                    "shape": _shape_text(spec),
                    "offset": str(offset),
                    "size": str(len(raw)),
                },
            )
        elif node.attrs:
            ET.SubElement(layer, "data", {key: _attribute_text(value) for key, value in node.attrs.items()})
        if node.inputs:
            input_element = ET.SubElement(layer, "input")
            for index, tensor in enumerate(node.inputs):
                _add_port(input_element, index, tensor)
        output_element = ET.SubElement(layer, "output")
        output_base = len(node.inputs)
        for index, tensor in enumerate(node.outputs):
            _add_port(output_element, output_base + index, tensor, names=tensor.name)

    result_base = len(nodes)
    for output_index, output in enumerate(graph.outputs):
        result_name = output.name or f"Result_{output_index}"
        layer = ET.SubElement(
            layers,
            "layer",
            {
                "id": str(result_base + output_index),
                "name": result_name,
                "type": "Result",
                "version": "opset1",
                "output_names": result_name,
            },
        )
        input_element = ET.SubElement(layer, "input")
        _add_port(input_element, 0, output)

    edges = ET.SubElement(root, "edges")
    for node in nodes:
        for input_index, tensor in enumerate(node.inputs):
            producer = tensor.producer
            ET.SubElement(
                edges,
                "edge",
                {
                    "from-layer": str(ids[producer]),
                    "from-port": str(len(producer.inputs) + tensor.output_index),
                    "to-layer": str(ids[node]),
                    "to-port": str(input_index),
                },
            )
    for output_index, tensor in enumerate(graph.outputs):
        producer = tensor.producer
        ET.SubElement(
            edges,
            "edge",
            {
                "from-layer": str(ids[producer]),
                "from-port": str(len(producer.inputs) + tensor.output_index),
                "to-layer": str(result_base + output_index),
                "to-port": "0",
            },
        )
    ET.SubElement(root, "rt_info")
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return SerializedIR(xml, bytes(weights), tuple(custom_nodes))
