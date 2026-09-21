/* Calibrated contiguous/static FP16 ACT entry, derived from marker-dims.c.
 * Metadata validation/general stride/layout handling are not established.
 * Reads the ACT input (already DPU add(x,bias)), not the graph's host input.
 */
static __attribute__((always_inline)) inline unsigned load32(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

void controlled_act(unsigned layerParams) {
    const unsigned char *params = (const unsigned char *)layerParams;
    unsigned rank = load32(params + 0xc);
    const unsigned char *dims = (const unsigned char *)load32(params + 0x10);
    if (rank == 0 || rank > 15 || !dims) return;
    unsigned count = 1;
    for (unsigned d = 0; d < rank; ++d) {
        unsigned dim = load32(dims + d * 4);
        if (dim == 0 || dim > 16 / count) return;
        count *= dim;
    }
    const __fp16 *in = (const __fp16 *)load32(params);
    __fp16 *out = (__fp16 *)load32(params + 0x28);
    for (unsigned i = 0; i < count; ++i)
        out[i] = in[i] + 1.0f;
}
