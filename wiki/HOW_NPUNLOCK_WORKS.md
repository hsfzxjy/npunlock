# How npunlock works

[Documentation index](README.md)

Intel's public NPU software accepts models and graphs, but it does not expose
a supported application workflow that turns arbitrary user C directly into an
ACT-SHAVE operation inside a native graph. `npunlock` therefore combines two
existing capabilities:

1. the installed Intel NPU driver knows how to build a valid native graph; and
2. MoviTools can compile C for the `3720xx` SHAVE target.

The project connects them without reimplementing Intel's graph compiler.

For a step-by-step account of how an ordinary neural-network graph is lowered
into DPU, ACT-SHAVE, DMA, and synchronization records, first see
[From neural network to machine binary](MODEL_TO_MACHINE_CODE.md). This page
focuses on the additional custom-kernel path provided by `npunlock`.

## Pipeline overview

```text
Python graph description
        |
        v
carrier graph in OpenVINO-format IR
        |
        v
installed Intel NPU driver compiles a native graph
        |
        +-------------------------------+
                                        |
user C                                  |
  -> MoviTools                          |
  -> validated SHAVE ELF                |
        |                               |
        +---------------+---------------+
                        v
             validated ACT code substitution
                        |
                        v
                patched native graph
                        |
                        v
              installed Intel NPU driver
                        |
                        v
                       NPU
```

OpenVINO-format IR is the serialization accepted by the driver compiler in
this path. The OpenVINO runtime and Python package are not loaded or required.

## 1. Represent the custom operation with a carrier

The Intel graph compiler does not know the semantics of user C. Before graph
compilation, `npunlock` temporarily represents the custom node with a normal
operator that the compiler already lowers to an ACT-SHAVE implementation.
This is the **carrier operator**.

For example:

```text
symbolic graph                 graph sent to Intel

input                          input
  |                              |
custom GELU                    Abs carrier
  |                              |
output                         output
```

The carrier determines the invocation environment: tensor arity, shape,
partitioning, memory placement, scheduling, and synchronization. It does not
determine the final custom computation.

Carrier choice is therefore a compatibility decision, not just a convenient
placeholder. The current Python examples use carriers whose compiled tensor
contracts have been validated for the exact graph shape and precision.

## 2. Ask the installed driver to build the graph

`ir2blob` serializes the graph as in-memory OpenVINO-format IR XML and weights,
then sends it to the Intel NPU Level Zero graph extension. The current NPU
driver selects its installed graph compiler, creates all native scheduling and
memory structures, and exports the complete native graph blob.

This step uses the machine's installed NPU driver. It does not use the old Lenovo
driver package from which MoviTools is extracted.

## 3. Compile the custom C separately

`shavecc` sends source bytes through three isolated MoviTools stages:

```text
C -> assembly -> relocatable object -> linked SHAVE ELF
```

The linked ELF is validated before use. The supported image has executable
`.text` at the expected address, an empty `.arg.data`, no undefined symbols,
and no unresolved relocation sections.

The Intel graph compiler never sees the custom C source.

## 4. Find a compatible ACT group

The native graph may contain several ACT operations and several invocations
per operation. `patchblob` parses the graph, follows its relocations, and
groups compatible invocation records by their observed positional identity.
It then validates tensor arity, element count, byte span, precision, dense
layout, CMX placement, and output separation.

The Python frontend normally maps the complete topological computational-node
sequence one-to-one to the discovered ACT groups. It can also combine several
consecutive groups for one large custom node when there is exactly one complete
mapping and the combined targets have matching arity, element width and
contract flags, consecutive invocation indices, and an element count that
exactly covers the declared output tensor. This is intentionally not a general
source-node-name lookup. Fused, optimized-away, inserted-conversion,
DPU-containing, or otherwise ambiguous graphs require explicit validated
targets or are rejected.

## 5. Substitute the executable image

`patchblob` extracts `.text` from the validated SHAVE ELF, appends it to the
graph's kernel-code section, and retargets only the selected ACT ranges. It
updates affected ELF file offsets and produces a preservation report.

The surrounding compiler-generated graph remains intact:

```text
preserved                        changed
---------                        -------
tensor metadata                  appended custom code image
kernel parameters               selected range extents
DMA tasks                        selected code relocations
barriers
DPU tasks
unselected ACT tasks
```

The parser fails closed when a structure or relocation does not match the
validated contract.

## 6. Execute and validate semantics

`graphinfer` loads the patched native graph through the installed Intel driver,
binds caller-provided tensors, executes with a finite deadline, and returns
owned output buffers. Python's `Program.run()` converts those buffers into
NumPy arrays.

A graph loading successfully does not prove that a new kernel computes the
right function. New kernels and carriers should always be checked against a
host reference.

## Confirmed versus inferred behavior

Confirmed behavior includes the complete unary FP16 path, a narrow
precision-preserved unary FP32 path, and a static dense FP16 binary carrier
with independently bound host inputs. The exact layouts used by these cases
are execution-tested.

Field names that are not available from a public vendor specification remain
described as observed values. The project does not generalize them to dynamic
shapes, arbitrary layouts, arbitrary graph structures, or other hardware
generations.

Continue with [Custom kernels](CUSTOM_KERNELS.md) for the C programming model,
or consult the [native graph ABI reference](../ABI/GRAPH_ELF.md) for byte-level
details.

[Back to documentation index](README.md)
