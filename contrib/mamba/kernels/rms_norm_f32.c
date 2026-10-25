#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define RMS_MAGIC 0x524d5331u
#define RMS_VERSION 1u
#define RMS_HEADER 16u
#define RMS_INVOCATION_WORDS 2048u
#define RMS_MAX_FEATURES 1536u

float sqrtf(float value);

static unsigned rms_word(const float *buffer, unsigned index) {
  union {
    float value;
    unsigned word;
  } item;
  item.value = buffer[index];
  return item.word;
}

static void rms_store_word(float *buffer, unsigned index, unsigned word) {
  union {
    float value;
    unsigned word;
  } item;
  item.word = word;
  buffer[index] = item.value;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *values;
  const float *weights;
  float *output;
  float sum_squares = 0.0f;
  float scale;
  unsigned count;
  unsigned index;
  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  values = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  weights = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  if (invocation.element_count != RMS_INVOCATION_WORDS || rms_word(values, 0u) != RMS_MAGIC ||
      rms_word(weights, 0u) != RMS_MAGIC || rms_word(values, 1u) != RMS_VERSION ||
      rms_word(weights, 1u) != RMS_VERSION) {
    return;
  }
  count = rms_word(values, 2u);
  if (count == 0u || count > RMS_MAX_FEATURES || count != rms_word(weights, 2u) ||
      RMS_HEADER + count > invocation.element_count || values[3u] <= 0.0f) {
    return;
  }
  for (index = 0u; index < count; ++index) {
    float value = values[RMS_HEADER + index];
    sum_squares += value * value;
  }
  scale = 1.0f / sqrtf(sum_squares / (float)count + values[3u]);
  rms_store_word(output, 0u, RMS_MAGIC);
  rms_store_word(output, 1u, RMS_VERSION);
  rms_store_word(output, 2u, count);
  for (index = 0u; index < count; ++index) {
    output[RMS_HEADER + index] = values[RMS_HEADER + index] * scale * weights[RMS_HEADER + index];
  }
}
