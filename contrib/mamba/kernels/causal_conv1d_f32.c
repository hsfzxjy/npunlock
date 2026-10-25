#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define CONV_MAGIC 0x434e5631u
#define CONV_VERSION 1u
#define CONV_HEADER 16u
#define CONV_INVOCATION_WORDS 2048u
#define CONV_WIDTH 4u
#define CONV_MAX_CHANNELS_PER_INVOCATION 384u

float expf(float value);

static unsigned conv_word(const float *buffer, unsigned index) {
  union {
    float value;
    unsigned word;
  } item;
  item.value = buffer[index];
  return item.word;
}

static void conv_store_word(float *buffer, unsigned index, unsigned word) {
  union {
    float value;
    unsigned word;
  } item;
  item.word = word;
  buffer[index] = item.value;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *dynamic;
  const float *model;
  float *output;
  unsigned channels;
  unsigned channel;
  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  dynamic = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  model = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  if (invocation.element_count != CONV_INVOCATION_WORDS || conv_word(dynamic, 0u) != CONV_MAGIC ||
      conv_word(model, 0u) != CONV_MAGIC || conv_word(dynamic, 1u) != CONV_VERSION ||
      conv_word(model, 1u) != CONV_VERSION) {
    return;
  }
  channels = conv_word(dynamic, 2u);
  if (channels > CONV_MAX_CHANNELS_PER_INVOCATION || channels != conv_word(model, 2u) ||
      CONV_HEADER + channels * (CONV_WIDTH + 1u) > invocation.element_count) {
    return;
  }
  conv_store_word(output, 0u, CONV_MAGIC);
  conv_store_word(output, 1u, CONV_VERSION);
  conv_store_word(output, 2u, channels);
  for (channel = 0u; channel < channels; ++channel) {
    const float *old_state = dynamic + CONV_HEADER + channels + channel * CONV_WIDTH;
    const float *weight = model + CONV_HEADER + channel * CONV_WIDTH;
    float *new_state = output + CONV_HEADER + channels + channel * CONV_WIDTH;
    float total;
    new_state[0u] = old_state[1u];
    new_state[1u] = old_state[2u];
    new_state[2u] = old_state[3u];
    new_state[3u] = dynamic[CONV_HEADER + channel];
    total = new_state[0u] * weight[0u] + new_state[1u] * weight[1u] +
            new_state[2u] * weight[2u] + new_state[3u] * weight[3u] +
            model[CONV_HEADER + channels * CONV_WIDTH + channel];
    output[CONV_HEADER + channel] = total / (1.0f + expf(-total));
  }
}
