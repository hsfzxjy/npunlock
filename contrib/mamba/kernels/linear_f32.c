#include <npunlock/npu3720_kernel.h>

#define LINEAR_MAGIC 0x4c494e31u
#define LINEAR_VERSION 1u
#define LINEAR_HEADER 16u
#define LINEAR_INVOCATION_WORDS 65536u
#define LINEAR_MAX_INPUTS 1536u

static unsigned linear_word(const float *buffer, unsigned index) {
  union {
    float value;
    unsigned word;
  } item;
  item.value = buffer[index];
  return item.word;
}

static void linear_store_word(float *buffer, unsigned index, unsigned word) {
  union {
    float value;
    unsigned word;
  } item;
  item.word = word;
  buffer[index] = item.value;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *activation;
  const float *weights;
  float *output;
  unsigned input_count;
  unsigned output_count;
  unsigned output_index;
  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  activation = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  weights = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  if (invocation.element_count != LINEAR_INVOCATION_WORDS ||
      linear_word(activation, 0u) != LINEAR_MAGIC || linear_word(weights, 0u) != LINEAR_MAGIC ||
      linear_word(activation, 1u) != LINEAR_VERSION ||
      linear_word(weights, 1u) != LINEAR_VERSION) {
    return;
  }
  input_count = linear_word(activation, 2u);
  output_count = linear_word(weights, 3u);
  if (input_count == 0u || input_count > LINEAR_MAX_INPUTS ||
      input_count != linear_word(weights, 2u) ||
      LINEAR_HEADER + input_count * output_count > invocation.element_count) {
    return;
  }
  linear_store_word(output, 0u, LINEAR_MAGIC);
  linear_store_word(output, 1u, LINEAR_VERSION);
  linear_store_word(output, 2u, output_count);
  for (output_index = 0u; output_index < output_count; ++output_index) {
    const float *row = weights + LINEAR_HEADER + output_index * input_count;
    float total = 0.0f;
    unsigned input_index;
    for (input_index = 0u; input_index < input_count; ++input_index) {
      total += activation[LINEAR_HEADER + input_index] * row[input_index];
    }
    output[LINEAR_HEADER + output_index] = total;
  }
}
