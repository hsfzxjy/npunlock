#ifndef NPUNLOCK_NPU3720_KERNEL_H
#define NPUNLOCK_NPU3720_KERNEL_H

/*
 * Helpers for the observed NPU3720 ACT invocation ABI.
 *
 * This is a target header for MoviTools, not a host C API.  Its names encode
 * the assumptions that pointers are carried in low 32-bit fields and tensor
 * records use the observed 0x28-byte NPU3720 layout.  Those assumptions must
 * not be reused for another NPU generation without separate validation.
 */

#define NPUNLOCK_NPU3720_ACT_ABI_MAX_RANK 15u
#define NPUNLOCK_NPU3720_ACT_ABI_TENSOR_RECORD_BYTES 0x28u
#define NPUNLOCK_NPU3720_ACT_ABI_DATA_PTR32_OFFSET 0x00u
#define NPUNLOCK_NPU3720_ACT_ABI_RANK_OFFSET 0x0cu
#define NPUNLOCK_NPU3720_ACT_ABI_DIMS_PTR32_OFFSET 0x10u

#define NPUNLOCK_NPU3720_ALWAYS_INLINE static __attribute__((always_inline)) inline

typedef struct npunlock_npu3720_act_abi_invocation {
  const unsigned char *params32;
  const unsigned char *dims32;
  unsigned rank;
  unsigned element_count;
} npunlock_npu3720_act_abi_invocation;

NPUNLOCK_NPU3720_ALWAYS_INLINE unsigned
npunlock_npu3720_act_abi_load_le_u32(const unsigned char *address) {
  return (unsigned)address[0] | ((unsigned)address[1] << 8) | ((unsigned)address[2] << 16) |
         ((unsigned)address[3] << 24);
}

NPUNLOCK_NPU3720_ALWAYS_INLINE int
npunlock_npu3720_act_abi_load_invocation32(unsigned layer_params32, unsigned maximum_elements,
                                           npunlock_npu3720_act_abi_invocation *invocation) {
  const unsigned char *params32;
  const unsigned char *dims32;
  unsigned rank;
  unsigned count = 1;
  unsigned dimension_index;

  if (layer_params32 == 0 || maximum_elements == 0 || invocation == 0) {
    return 0;
  }
  params32 = (const unsigned char *)layer_params32;
  rank = npunlock_npu3720_act_abi_load_le_u32(params32 + NPUNLOCK_NPU3720_ACT_ABI_RANK_OFFSET);
  dims32 = (const unsigned char *)npunlock_npu3720_act_abi_load_le_u32(
      params32 + NPUNLOCK_NPU3720_ACT_ABI_DIMS_PTR32_OFFSET);
  if (rank == 0 || rank > NPUNLOCK_NPU3720_ACT_ABI_MAX_RANK || dims32 == 0) {
    return 0;
  }
  for (dimension_index = 0; dimension_index < rank; ++dimension_index) {
    unsigned dimension = npunlock_npu3720_act_abi_load_le_u32(dims32 + dimension_index * 4u);
    if (dimension == 0 || dimension > maximum_elements / count) {
      return 0;
    }
    count *= dimension;
  }
  invocation->params32 = params32;
  invocation->dims32 = dims32;
  invocation->rank = rank;
  invocation->element_count = count;
  return 1;
}

#define NPUNLOCK_NPU3720_ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layer_params32, maximum_elements,     \
                                                             invocation)                           \
  do {                                                                                             \
    if (!npunlock_npu3720_act_abi_load_invocation32((layer_params32), (maximum_elements),          \
                                                    &(invocation))) {                              \
      return;                                                                                      \
    }                                                                                              \
  } while (0)

#define NPUNLOCK_NPU3720_ACT_ABI_TENSOR_PTR32(type, invocation, record_index)                      \
  ((type *)npunlock_npu3720_act_abi_load_le_u32((invocation).params32 +                            \
                                                (unsigned)(record_index) *                         \
                                                    NPUNLOCK_NPU3720_ACT_ABI_TENSOR_RECORD_BYTES + \
                                                NPUNLOCK_NPU3720_ACT_ABI_DATA_PTR32_OFFSET))

#define NPUNLOCK_NPU3720_ACT_ABI_INPUT_PTR32(type, invocation, input_index)                        \
  NPUNLOCK_NPU3720_ACT_ABI_TENSOR_PTR32(type, invocation, input_index)

#define NPUNLOCK_NPU3720_ACT_ABI_OUTPUT_PTR32(type, invocation, input_count)                       \
  NPUNLOCK_NPU3720_ACT_ABI_TENSOR_PTR32(type, invocation, input_count)

/*
 * Define the external symbols required by the observed NPU3720 mlibm.a
 * dependency closure.  strtof and __truncdfsf2 are deterministic link stubs
 * for the tested tanhf path, not general implementations.  __fixsfdi provides
 * the binary32 conversion used by that path without another compiler helper.
 * Invoke this macro exactly once in a translation unit that uses mlibm.a.
 */
#define NPUNLOCK_NPU3720_MLIBM_DEFINE_LINK_COMPAT()                                                \
  float strtof(const char *text, char **end_pointer) {                                             \
    (void)text;                                                                                    \
    (void)end_pointer;                                                                             \
    return 0.0f;                                                                                   \
  }                                                                                                \
  float __truncdfsf2(double value) {                                                               \
    (void)value;                                                                                   \
    return 0.0f;                                                                                   \
  }                                                                                                \
  long long __fixsfdi(float value) {                                                               \
    union {                                                                                        \
      float f;                                                                                     \
      unsigned u;                                                                                  \
    } bits = {value};                                                                              \
    unsigned sign = bits.u >> 31;                                                                  \
    unsigned fraction = bits.u & 0x007fffffu;                                                      \
    int exponent = (int)((bits.u >> 23) & 0xffu) - 127;                                            \
    unsigned long long mantissa;                                                                   \
    unsigned long long integer_value;                                                              \
    if (exponent < 0) {                                                                            \
      return 0;                                                                                    \
    }                                                                                              \
    mantissa = (unsigned long long)(fraction | 0x00800000u);                                       \
    integer_value = exponent <= 23 ? mantissa >> (23 - exponent) : mantissa << (exponent - 23);    \
    return sign ? -(long long)integer_value : (long long)integer_value;                            \
  }

#endif
