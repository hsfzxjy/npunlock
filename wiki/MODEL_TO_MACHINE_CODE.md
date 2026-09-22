# How a neural network becomes an NPU machine binary

[Documentation index](README.md)

A neural network is not translated into one long stream of instructions. An
Intel NPU contains several kinds of engines, so its compiler turns a graph into
a coordinated package of compute tasks, data movement, memory descriptions,
software kernels, and synchronization. That package is the **native graph
binary** loaded by the driver.

This page follows that transformation step by step. Some compiler decisions are
not publicly documented; where their exact implementation is unknown, this
page describes only the behavior visible at the input and output boundaries.

## The complete path

```text
Python graph or model
        |
        | serialize operations, tensors, shapes, constants
        v
OpenVINO-format IR XML + weights
        |
        | Level Zero NPU graph extension
        v
Intel NPU user-mode driver and installed graph compiler
        |
        | parse, optimize, map, partition, schedule, allocate, encode
        v
native graph binary
        |
        | load, bind input/output buffers, initialize, execute
        v
Intel NPU firmware and hardware
        |
        +--> DPU tensor work
        +--> ACT-SHAVE software kernels
        +--> DMA transfers and barriers
```

`npunlock` does not replace the Intel graph compiler. It supplies the graph in
the format accepted by the installed driver, receives the resulting native
binary, and—only for selected custom operations—substitutes validated
ACT-SHAVE code before the graph is loaded again.

## Components involved

| Component | Responsibility |
| --- | --- |
| Python frontend | Describes operations, tensor shapes, types, constants, inputs, outputs, and custom nodes. |
| IR serializer | Converts that symbolic graph to OpenVINO-format XML and a weights buffer without loading the OpenVINO runtime. |
| `ir2blob` | Sends the in-memory IR package to a bounded worker and returns the complete native graph binary. |
| Level Zero loader | Locates the selected Intel NPU driver and exposes its NPU graph extension. |
| Intel NPU user-mode driver (UMD) | Provides the graph API and the installed graph compiler used for the current device. |
| Intel graph compiler | Lowers the model graph into device-specific compute, movement, memory, and synchronization records. |
| Native graph binary | Carries the compiled tasks, metadata, relocations, parameters, and ACT-SHAVE code needed to recreate the graph. |
| MoviTools | Separately compiles user C into a SHAVE ELF for a custom operation; it does not compile the neural-network graph. |
| `patchblob` | Validates the native graph and substitutes custom code into selected compatible ACT ranges. |
| `graphinfer` | Reloads the native graph, binds caller tensors, submits initialization and execution commands, and returns outputs. |
| Driver, firmware, and NPU | Relocate and execute the graph across DPU, ACT-SHAVE, DMA, and synchronization resources. |

## Step 1: describe the graph

At the highest level, a model is a directed graph. Nodes describe operations;
edges describe tensors flowing between them. Shapes, element types, constant
values, and graph inputs and outputs are part of this description.

With the `npunlock` Python API, ordinary operations and custom operations share
the same symbolic graph. Before graph compilation, a custom operation is
represented by a known **carrier operation** because Intel's compiler cannot
compile the semantics of arbitrary user C.

```text
user graph                    graph presented to Intel

input                         input
  |                             |
custom GELU                   Abs carrier
  |                             |
output                        output
```

The carrier gives Intel's compiler a real operation from which it can build a
compatible tensor and scheduling environment. Its executable implementation
is replaced later.

## Step 2: serialize the graph as portable IR

The Python serializer writes two in-memory buffers:

- XML describing graph nodes, ports, edges, shapes, types, and attributes; and
- a weights buffer containing constants referenced by the XML.

This is OpenVINO-format IR, but creating it does not require the OpenVINO Python
package or runtime. It is a graph interchange format at this boundary, not the
device machine binary.

## Step 3: enter the current Intel NPU software stack

`ir2blob` passes the XML, weights, and build flags to a private worker process.
The worker loads the system Level Zero loader, selects an NPU device, obtains
the driver's graph-extension function table, and verifies that the device
advertises the `NGRAPH_LITE` input format.

The worker packages the IR entirely in memory and submits it through the
graph extension's `Create2` path. This call crosses into the current Intel NPU
user-mode driver and its installed graph compiler. The legacy Lenovo package
used to obtain MoviTools is not involved in this step.

## Step 4: lower the graph for the device

The Intel graph compiler performs the device-specific transformation. The exact
pass sequence and algorithms are not a public contract, but the exported graph
shows the responsibilities that compilation resolves:

1. **Parse and validate the graph.** Operations, tensor connections, shapes,
   precisions, constants, and requested compiler options are checked.
2. **Transform and optimize it.** Operations may be legalized, converted,
   combined, removed, or partitioned. Consequently, a source node does not
   necessarily correspond to one native task.
3. **Choose execution resources.** Regular tensor work can become DPU tasks;
   programmable operations can become ACT-SHAVE invocations. Data transfers
   become DMA tasks.
4. **Partition the work.** One logical operation may be split into multiple
   invocations or hardware work items according to shape, tiling, and resource
   decisions.
5. **Plan tensor storage and movement.** The compiler defines tensor layouts,
   working-memory placement, parameters, and the transfers between producers
   and consumers.
6. **Schedule and synchronize.** Tasks receive ordering relationships and
   barriers so compute and data movement happen safely.
7. **Encode device records and code.** The compiler emits DPU configuration,
   ACT-SHAVE code and parameters, DMA records, barrier records, metadata, and
   relocations connecting those pieces.

The broad responsibilities above are confirmed by the records present in
exported native graphs. The compiler's private optimization heuristics,
intermediate representations, scheduling algorithms, and complete instruction
encodings are not claimed or required by `npunlock`.

## Step 5: export the native graph binary

The worker asks the graph extension for the complete native binary and copies
it before destroying the temporary graph object. It then reloads that binary
once through the same driver as a basic compatibility check.

On the confirmed NPU3720 path, the exported object is an ELF64 container. It is
not a conventional CPU executable and it is not only machine-code bytes. Its
observed sections include:

```text
ACT-SHAVE code and optional data
per-invocation kernel parameters
ACT kernel ranges and invocation records
DPU invariant and variant records
DMA tasks
barrier and synchronization records
top-level mapped-inference data
graph metadata, symbols, and relocations
```

Different engines consume different representations. ACT-SHAVE sections
contain programmable instruction bytes. DPU, DMA, and barrier sections contain
device command/configuration records rather than one shared CPU-like instruction
stream. Relocations connect addresses and references after the graph is loaded.

For the byte-level observed structure, see the
[native graph ELF ABI](../ABI/GRAPH_ELF.md).

## Step 6: insert a custom kernel, when requested

Custom C follows a separate compiler path:

```text
C source -> MoviTools compiler -> assembly -> object -> linked SHAVE ELF
```

`shavecc` validates that ELF. `patchblob` then locates the compatible carrier
ACT group, appends the custom executable image to the graph's kernel-code
section, and retargets only the selected code ranges and relocations. Tensor
placement, scheduling, DMA, barriers, and unselected compute tasks remain those
created by Intel's compiler.

This is why custom code cannot be inserted into an arbitrary graph position:
the carrier's compiler-generated invocation environment must match the custom
kernel's validated tensor contract.

## Step 7: load and execute the binary

`graphinfer` submits the resulting blob to the graph extension in native format.
It queries the compiled graph's argument metadata, allocates host-visible Level
Zero buffers, validates and binds every input and output, and records graph
initialization and execution commands in a Level Zero command list.

The driver and firmware then prepare the native records for the device and
coordinate DPU work, ACT-SHAVE invocations, DMA transfers, and barriers. After
bounded completion, `graphinfer` copies all output tensors into caller-owned
results.

Loading and executing a binary demonstrates structural compatibility; it does
not by itself prove that a custom kernel computes the intended function.
Examples and new kernels therefore compare NPU output with a host-side oracle.

## What is known and what remains internal

The following boundaries are directly exercised by `npunlock`:

- IR XML and weights enter through the installed driver's `NGRAPH_LITE` graph
  interface;
- a complete native ELF graph can be exported, saved, reloaded, and executed;
- its observed sections describe DPU, ACT, DMA, barrier, parameter, and
  relocation data;
- validated ACT code can be replaced while the rest of the graph is preserved;
  and
- the resulting controlled graphs execute correctly on the confirmed NPU3720
  configuration.

The following remain vendor-internal and should not be inferred from this
overview:

- the graph compiler's exact sequence of internal passes;
- all rules used to fuse, split, place, or schedule operations;
- a universal mapping from model nodes to native records;
- the complete DPU, ACT-SHAVE, DMA, or firmware ABI; and
- compatibility with untested compiler versions or NPU generations.

Continue with [How npunlock works](HOW_NPUNLOCK_WORKS.md) for the custom-kernel
integration path, or [Intel NPU architecture](INTEL_NPU_ARCHITECTURE.md) for a
short introduction to the hardware roles.

[Back to documentation index](README.md)
