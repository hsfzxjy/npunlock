# Custom kernels

[Documentation index](README.md)

Custom kernels are C functions compiled for the NPU3720 ACT-SHAVE processors.
They run inside an invocation environment created by Intel's graph compiler.
That environment supplies tensor pointers and dimensions; the kernel supplies
the computation.

The contract described here is intentionally narrow and execution-tested. It
is not a general Intel SHAVE ABI.

## Entry point

The default entry function is:

```c
void controlled_act(unsigned layerParams);
```

`layerParams` is the low 32-bit address of the current invocation's parameter
block. Pointer-bearing fields should be loaded explicitly as little-endian
32-bit values:

```c
static __attribute__((always_inline)) inline unsigned load32(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}
```

The dimensions array contains one 32-bit value per rank dimension. Always
validate rank and dimension multiplication before entering the element loop.

## Unary FP16 example

This complete kernel adds one to every element in its invocation-local chunk:

```c
static __attribute__((always_inline)) inline unsigned load32(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

void controlled_act(unsigned layerParams) {
    const unsigned char *params = (const unsigned char *)layerParams;
    unsigned rank = load32(params + 0x0c);
    const unsigned char *dims = (const unsigned char *)load32(params + 0x10);
    unsigned count = 1;

    if (rank == 0 || rank > 15 || !dims) return;
    for (unsigned d = 0; d < rank; ++d) {
        unsigned dim = load32(dims + d * 4);
        if (dim == 0 || dim > 2048u / count) return;
        count *= dim;
    }

    const __fp16 *input = (const __fp16 *)load32(params + 0x00);
    __fp16 *output = (__fp16 *)load32(params + 0x28);
    for (unsigned i = 0; i < count; ++i) {
        output[i] = (__fp16)((float)input[i] + 1.0f);
    }
}
```

The capacity guard belongs to the kernel source. Choose it for the exact
validated carrier; do not silently assume that every invocation has the same
size.

## Tensor descriptor positions

The validated parameter records are 0x28 bytes each:

```text
unary:
  input  at layerParams + 0x00
  output at layerParams + 0x28

binary:
  input A at layerParams + 0x00
  input B at layerParams + 0x28
  output  at layerParams + 0x50
```

Within each record, the fields used by the examples are:

```text
+0x00  low 32 bits of data address
+0x0c  rank
+0x10  low 32 bits of dimensions-array address
```

`patchblob` validates the rest of the observed contract before installation:
static dimensions, dense strides, supported precision, matching element
counts and spans, CMX placement, and a disjoint output.

## Supported precision

Static dense FP16 is the primary custom-kernel contract. Both unary and a
narrow two-input layout have been executed successfully.

FP32 custom execution is limited to the validated unary accuracy-mode path.
The Python frontend adds the required precision-preservation metadata and
compiler flag automatically for an FP32 custom node. This does not establish
general FP32 carriers or FP32 multi-input support.

Although the symbolic IR serializer accepts additional data-type names, that
does not make them valid custom-kernel or `Program.run()` contracts.

## Invocation-local memory

The Intel compiler partitions tensors into ACT invocations. The descriptor
seen by the kernel describes the current chunk, not necessarily the full
graph tensor.

Validated FP16 carriers have advertised chunks from 8 through 2048 elements.
A kernel has read the full 4096-byte span of a 2048-element chunk. This is a
confirmed lower bound for invocation-local reach, not a hardware maximum and
not permission to access another invocation's memory.

Operations requiring neighboring values must define behavior at each chunk
boundary. Graph-global coordinates and cross-chunk halos are not available in
the current contract.

## Carrier selection

Python custom nodes name a carrier explicitly:

```python
y = npu.custom(
    x,
    source=kernel_c,
    carrier="Abs",
    _shape=x.shape,
    _dtype=x.dtype,
    _name="y",
)
```

The carrier's compiled arity and tensor layout must match the C entry. `Abs`
is used by the validated unary examples. The validated two-input example uses
`Maximum` for one exact static FP16 graph.

Automatic positional mapping is accepted only when the complete graph maps
one-to-one to validated ACT groups. See
[How npunlock works](HOW_NPUNLOCK_WORKS.md#4-find-a-compatible-act-group).

## Math functions and linked data

MoviTools' `mlibm.a` supplies functions such as `tanhf`. `shavecc` links the
archive with section garbage collection so only reachable code remains.

The resulting kernel must be self-contained in `.text`. The current validator
requires `.arg.data` to be empty, so mutable globals, retained writable library
state, unresolved relocations, and general helper runtimes are unsupported.
Read-only constants are usable when the linker places them inside the final
`.text` image.

## Compiler definitions and linker script

`npu.compile(..., definitions=("NAME=123",))` forwards a constrained set of
uppercase decimal definitions. The default NPU3720 linker script is embedded
in the native library. Supplying another script is an advanced override and
does not relax ELF validation.

## Runnable examples

- [FP16 GELU](../examples/example_gelu.py)
- [FP32 GELU](../examples/example_gelu_f32.py)
- [Multi-layer, two-input custom kernel](../examples/example_multilayer_multi_input.py)

Each example includes a NumPy reference. Successful graph submission is not a
correctness oracle; use equivalent host validation for every new kernel.

For the complete byte-level entry and ELF contract, see
[ACT kernel ELF ABI](../ABI/KERNEL_ELF.md).

[Back to documentation index](README.md)
