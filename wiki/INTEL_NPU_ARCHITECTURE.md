# Intel NPU architecture for npunlock users

[Documentation index](README.md)

An NPU is a processor specialized for running neural-network workloads with
high energy efficiency. Intel Core Ultra systems place the NPU alongside the
CPU and GPU so models can run locally without using the CPU for every tensor
operation.

This page introduces only the hardware concepts needed to understand
`npunlock`. It is not a complete microarchitecture description.

## Two kinds of compute

At a high level, the tested Intel NPU exposes two relevant styles of compute:

```text
                       Intel NPU

            +---------------------------+
            | DPU / tensor engines      |
            |                           |
            | dense, regular tensor     |
            | computation               |
            +-------------+-------------+
                          |
                       tensors
                          |
            +-------------v-------------+
            | ACT-SHAVE processors      |
            |                           |
            | programmable software     |
            | operations                |
            +---------------------------+
```

The DPU path is suited to regular operations such as convolution and matrix
multiplication. ACT-SHAVE processors execute software kernels used for
operations that need a programmable implementation.

Most applications never choose between these resources. They submit a graph,
and Intel's NPU compiler decides where each operation runs.

## Why ACT-SHAVE matters

Native NPU graphs already contain ACT-SHAVE machine code. The driver compiler
places that code in the graph together with tensor addresses, work
partitions, DMA tasks, and synchronization.

The normal public software path exposes the NPU at the graph or model level,
not as a low-level kernel-programming target. There is no ordinary application
workflow equivalent to:

```text
my_kernel.c -> compile -> schedule directly on ACT-SHAVE
```

`npunlock` supplies a narrow version of that missing path. It compiles user C
for ACT-SHAVE and integrates the code into a compatible graph environment
that Intel's compiler has already constructed.

## Graph compilation still matters

A native graph is not merely a list of operator implementations. It also
describes:

- where tensors live;
- how tensors move between memory and compute resources;
- how work is divided into invocations;
- task ordering and barriers; and
- the parameters passed to each kernel.

`npunlock` does not try to replace this machinery. The installed Intel NPU
driver still compiles the graph and executes it. `npunlock` preserves that
environment while substituting the executable code for a validated ACT
operation.

## Current hardware boundary

The confirmed target is Meteor Lake / NPU3720, using the MoviTools `3720xx`
target. Similar names or related Intel NPU generations do not imply
compatibility. See [Current limitations](LIMITATIONS.md) for the complete
support boundary.

For the end-to-end software flow, continue with
[How npunlock works](HOW_NPUNLOCK_WORKS.md).

[Back to documentation index](README.md)
