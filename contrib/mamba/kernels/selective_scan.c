#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define MAMBA_MAGIC 0x4d414d31u
#define MAMBA_VERSION 1u
#define MAMBA_HEADER_WORDS 16u
#define MAMBA_MAX_STATE 16u
#define MAMBA_MAX_SEQUENCE 16u
#define MAMBA_INVOCATION_WORDS 32768u
#define MAMBA_STATUS_OK 0u
#define MAMBA_STATUS_BAD_HEADER 1u
#define MAMBA_STATUS_BAD_SHAPE 2u
#define MAMBA_STATUS_BAD_CAPACITY 3u

float expf(float value);
float logf(float value);
void *memcpy(void *destination, const void *source, unsigned size);

static float mamba_softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return logf(1.0f + expf(value));
}

static float mamba_silu(float value) { return value / (1.0f + expf(-value)); }

static unsigned mamba_load_word(const float *buffer, unsigned index) {
  union {
    float value;
    unsigned word;
  } item;
  item.value = buffer[index];
  return item.word;
}

static void mamba_store_word(float *buffer, unsigned index, unsigned word) {
  union {
    float value;
    unsigned word;
  } item;
  item.word = word;
  buffer[index] = item.value;
}

static void mamba_fail(float *output, unsigned status) {
  mamba_store_word(output, 0u, MAMBA_MAGIC);
  mamba_store_word(output, 1u, MAMBA_VERSION);
  mamba_store_word(output, 2u, status);
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *dynamic_input;
  const float *model_input;
  float *output;
  unsigned channels;
  unsigned state_size;
  unsigned sequence_length;
  unsigned dynamic_stride;
  unsigned model_stride;
  unsigned output_stride;
  unsigned bc_words;
  unsigned dynamic_required;
  unsigned model_required;
  unsigned output_required;
  unsigned channel;

  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  dynamic_input = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  model_input = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  if (dynamic_input == 0 || model_input == 0 || output == 0) {
    return;
  }
  if (invocation.element_count != MAMBA_INVOCATION_WORDS ||
      mamba_load_word(dynamic_input, 0u) != MAMBA_MAGIC ||
      mamba_load_word(model_input, 0u) != MAMBA_MAGIC ||
      mamba_load_word(dynamic_input, 1u) != MAMBA_VERSION ||
      mamba_load_word(model_input, 1u) != MAMBA_VERSION) {
    mamba_fail(output, MAMBA_STATUS_BAD_HEADER);
    return;
  }

  channels = mamba_load_word(dynamic_input, 2u);
  state_size = mamba_load_word(dynamic_input, 3u);
  sequence_length = mamba_load_word(dynamic_input, 4u);
  if (channels != mamba_load_word(model_input, 2u) ||
      state_size != mamba_load_word(model_input, 3u) || state_size == 0u ||
      state_size > MAMBA_MAX_STATE || sequence_length == 0u ||
      sequence_length > MAMBA_MAX_SEQUENCE) {
    mamba_fail(output, MAMBA_STATUS_BAD_SHAPE);
    return;
  }

  dynamic_stride = sequence_length * 3u + state_size;
  model_stride = state_size + 2u;
  output_stride = sequence_length + state_size;
  bc_words = sequence_length * state_size * 2u;
  dynamic_required = MAMBA_HEADER_WORDS + bc_words + channels * dynamic_stride;
  model_required = MAMBA_HEADER_WORDS + channels * model_stride;
  output_required = MAMBA_HEADER_WORDS + channels * output_stride;
  if (dynamic_required > invocation.element_count || model_required > invocation.element_count ||
      output_required > invocation.element_count) {
    mamba_fail(output, MAMBA_STATUS_BAD_CAPACITY);
    return;
  }

  mamba_store_word(output, 0u, MAMBA_MAGIC);
  mamba_store_word(output, 1u, MAMBA_VERSION);
  mamba_store_word(output, 2u, MAMBA_STATUS_OK);
  mamba_store_word(output, 3u, channels);
  mamba_store_word(output, 4u, state_size);
  mamba_store_word(output, 5u, sequence_length);

  for (channel = 0u; channel < channels; ++channel) {
    const float *dynamic_record =
        dynamic_input + MAMBA_HEADER_WORDS + bc_words + channel * dynamic_stride;
    const float *model_record = model_input + MAMBA_HEADER_WORDS + channel * model_stride;
    float *output_record = output + MAMBA_HEADER_WORDS + channel * output_stride;
    const float *u = dynamic_record;
    const float *delta = u + sequence_length;
    const float *z = delta + sequence_length;
    const float *initial_state = z + sequence_length;
    const float *a = model_record;
    float skip = model_record[state_size];
    float delta_bias = model_record[state_size + 1u];
    float state[MAMBA_MAX_STATE];
    unsigned state_index;
    unsigned time_index;

    for (state_index = 0u; state_index < state_size; ++state_index) {
      state[state_index] = initial_state[state_index];
    }
    for (time_index = 0u; time_index < sequence_length; ++time_index) {
      float transformed_delta = mamba_softplus(delta[time_index] + delta_bias);
      float input_value = u[time_index];
      float result = 0.0f;
      const float *b = dynamic_input + MAMBA_HEADER_WORDS + time_index * state_size;
      const float *c = dynamic_input + MAMBA_HEADER_WORDS + sequence_length * state_size +
                       time_index * state_size;
      for (state_index = 0u; state_index < state_size; ++state_index) {
        state[state_index] = expf(transformed_delta * a[state_index]) * state[state_index] +
                             transformed_delta * b[state_index] * input_value;
        result += c[state_index] * state[state_index];
      }
      output_record[time_index] = (result + skip * input_value) * mamba_silu(z[time_index]);
    }
    for (state_index = 0u; state_index < state_size; ++state_index) {
      output_record[sequence_length + state_index] = state[state_index];
    }
  }
}

void *memcpy(void *destination, const void *source, unsigned size) {
  unsigned char *destination_bytes = (unsigned char *)destination;
  const unsigned char *source_bytes = (const unsigned char *)source;
  unsigned index;
  for (index = 0u; index < size; ++index) {
    destination_bytes[index] = source_bytes[index];
  }
  return destination;
}
