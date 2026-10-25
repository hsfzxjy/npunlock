#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

#define LINEAR_MAGIC 0x4c494e31u
#define LINEAR_VERSION 1u
#define LINEAR_HEADER 16u
#define LINEAR_INVOCATION_WORDS 65536u
#define LINEAR_MAX_INPUTS 1536u
#define LINEAR_FLAG_RESIDUAL 1u
#define LINEAR_FLAG_RMS_NORM 2u
#define LINEAR_FLAG_CONVOLUTION 4u
#define LINEAR_CONV_WIDTH 4u
#define LINEAR_STATE_OUTPUT_OFFSET 4096u
#define LINEAR_CONV_OUTPUT_OFFSET 12288u
#define MAMBA_MAGIC 0x4d414d31u
#define MAMBA_VERSION 1u
#define MAMBA_HEADER 16u
#define MAMBA_MAX_STATE 16u
#define MAMBA_MAX_SEQUENCE 16u

float expf(float value);
float logf(float value);
float sqrtf(float value);
void *memcpy(void *destination, const void *source, unsigned size);

typedef float linear_float4 __attribute__((ext_vector_type(4)));

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

static float linear_dot(const float *left, const float *right, unsigned count) {
  float total = 0.0f;
  unsigned index = 0u;
  if ((((unsigned)left | (unsigned)right) & 15u) == 0u) {
    linear_float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
    const linear_float4 *left4 = (const linear_float4 *)left;
    const linear_float4 *right4 = (const linear_float4 *)right;
    for (; index + 4u <= count; index += 4u) {
      sums += left4[index / 4u] * right4[index / 4u];
    }
    total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
  for (; index < count; ++index) {
    total += left[index] * right[index];
  }
  return total;
}

static void linear_dot_pair(const float *left0, const float *left1, const float *right,
                            unsigned count, float *total0, float *total1) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  unsigned index = 0u;
  if ((((unsigned)left0 | (unsigned)left1 | (unsigned)right) & 15u) == 0u) {
    linear_float4 sums0 = {0.0f, 0.0f, 0.0f, 0.0f};
    linear_float4 sums1 = {0.0f, 0.0f, 0.0f, 0.0f};
    const linear_float4 *left04 = (const linear_float4 *)left0;
    const linear_float4 *left14 = (const linear_float4 *)left1;
    const linear_float4 *right4 = (const linear_float4 *)right;
    for (; index + 4u <= count; index += 4u) {
      linear_float4 weight = right4[index / 4u];
      sums0 += left04[index / 4u] * weight;
      sums1 += left14[index / 4u] * weight;
    }
    sum0 = (sums0[0] + sums0[1]) + (sums0[2] + sums0[3]);
    sum1 = (sums1[0] + sums1[1]) + (sums1[2] + sums1[3]);
  }
  for (; index < count; ++index) {
    float weight = right[index];
    sum0 += left0[index] * weight;
    sum1 += left1[index] * weight;
  }
  *total0 = sum0;
  *total1 = sum1;
}

static float linear_rms_dot(const float *values, const float *norm_weights, const float *row,
                            unsigned count, float rms_scale) {
  float total = 0.0f;
  unsigned index = 0u;
  if ((((unsigned)values | (unsigned)norm_weights | (unsigned)row) & 15u) == 0u) {
    linear_float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
    const linear_float4 *values4 = (const linear_float4 *)values;
    const linear_float4 *norm_weights4 = (const linear_float4 *)norm_weights;
    const linear_float4 *row4 = (const linear_float4 *)row;
    for (; index + 4u <= count; index += 4u) {
      sums += (values4[index / 4u] * (norm_weights4[index / 4u] * rms_scale)) *
              row4[index / 4u];
    }
    total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
  for (; index < count; ++index) {
    float value = values[index];
    value *= rms_scale * norm_weights[index];
    total += value * row[index];
  }
  return total;
}

static void linear_rms_dot_pair(const float *values0, const float *values1,
                                const float *norm_weights, const float *row, unsigned count,
                                float rms_scale0, float rms_scale1, float *total0,
                                float *total1) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  unsigned index = 0u;
  if ((((unsigned)values0 | (unsigned)values1 | (unsigned)norm_weights | (unsigned)row) & 15u) ==
      0u) {
    linear_float4 sums0 = {0.0f, 0.0f, 0.0f, 0.0f};
    linear_float4 sums1 = {0.0f, 0.0f, 0.0f, 0.0f};
    const linear_float4 *values04 = (const linear_float4 *)values0;
    const linear_float4 *values14 = (const linear_float4 *)values1;
    const linear_float4 *norm_weights4 = (const linear_float4 *)norm_weights;
    const linear_float4 *row4 = (const linear_float4 *)row;
    for (; index + 4u <= count; index += 4u) {
      linear_float4 norm_weight = norm_weights4[index / 4u];
      linear_float4 weight = row4[index / 4u];
      sums0 += (values04[index / 4u] * (norm_weight * rms_scale0)) * weight;
      sums1 += (values14[index / 4u] * (norm_weight * rms_scale1)) * weight;
    }
    sum0 = (sums0[0] + sums0[1]) + (sums0[2] + sums0[3]);
    sum1 = (sums1[0] + sums1[1]) + (sums1[2] + sums1[3]);
  }
  for (; index < count; ++index) {
    float weight = row[index];
    sum0 += (values0[index] * (norm_weights[index] * rms_scale0)) * weight;
    sum1 += (values1[index] * (norm_weights[index] * rms_scale1)) * weight;
  }
  *total0 = sum0;
  *total1 = sum1;
}

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

static void mamba_fail(float *output, unsigned status) {
  linear_store_word(output, 0u, MAMBA_MAGIC);
  linear_store_word(output, 1u, MAMBA_VERSION);
  linear_store_word(output, 2u, status);
}

static void mamba_execute(act_abi_invocation invocation, const float *dynamic_input,
                          const float *model_input, float *output) {
  unsigned channels = linear_word(dynamic_input, 2u);
  unsigned state_size = linear_word(dynamic_input, 3u);
  unsigned sequence_length = linear_word(dynamic_input, 4u);
  unsigned dynamic_stride;
  unsigned model_stride;
  unsigned output_stride;
  unsigned bc_words;
  unsigned channel;
  if (linear_word(model_input, 0u) != MAMBA_MAGIC ||
      linear_word(dynamic_input, 1u) != MAMBA_VERSION ||
      linear_word(model_input, 1u) != MAMBA_VERSION) {
    mamba_fail(output, 1u);
    return;
  }
  if (channels != linear_word(model_input, 2u) || state_size != linear_word(model_input, 3u) ||
      state_size == 0u || state_size > MAMBA_MAX_STATE || sequence_length == 0u ||
      sequence_length > MAMBA_MAX_SEQUENCE) {
    mamba_fail(output, 2u);
    return;
  }
  dynamic_stride = sequence_length * 3u + state_size;
  model_stride = state_size + 2u;
  output_stride = sequence_length + state_size;
  bc_words = sequence_length * state_size * 2u;
  if (MAMBA_HEADER + bc_words + channels * dynamic_stride > invocation.element_count ||
      MAMBA_HEADER + channels * model_stride > invocation.element_count ||
      MAMBA_HEADER + channels * output_stride > invocation.element_count) {
    mamba_fail(output, 3u);
    return;
  }
  linear_store_word(output, 0u, MAMBA_MAGIC);
  linear_store_word(output, 1u, MAMBA_VERSION);
  linear_store_word(output, 2u, 0u);
  linear_store_word(output, 3u, channels);
  linear_store_word(output, 4u, state_size);
  linear_store_word(output, 5u, sequence_length);
  for (channel = 0u; channel < channels; ++channel) {
    const float *dynamic_record =
        dynamic_input + MAMBA_HEADER + bc_words + channel * dynamic_stride;
    const float *model_record = model_input + MAMBA_HEADER + channel * model_stride;
    float *output_record = output + MAMBA_HEADER + channel * output_stride;
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
      const float *b = dynamic_input + MAMBA_HEADER + time_index * state_size;
      const float *c = dynamic_input + MAMBA_HEADER + sequence_length * state_size +
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

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *activation;
  const float *weights;
  float *output;
  unsigned input_count;
  unsigned output_count;
  unsigned flags;
  unsigned sequence_length;
  unsigned auxiliary_offset;
  unsigned residual_offset = 0u;
  unsigned rms_weight_offset = 0u;
  unsigned conv_state_offset = 0u;
  unsigned conv_weight_offset = 0u;
  unsigned conv_bias_offset = 0u;
  unsigned weight_input_rows;
  float rms_scales[MAMBA_MAX_SEQUENCE];
  unsigned output_index;
  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  activation = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  weights = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  if (linear_word(activation, 0u) == MAMBA_MAGIC) {
    mamba_execute(invocation, activation, weights, output);
    return;
  }
  if (invocation.element_count != LINEAR_INVOCATION_WORDS ||
      linear_word(activation, 0u) != LINEAR_MAGIC || linear_word(weights, 0u) != LINEAR_MAGIC ||
      linear_word(activation, 1u) != LINEAR_VERSION ||
      linear_word(weights, 1u) != LINEAR_VERSION) {
    return;
  }
  input_count = linear_word(activation, 2u);
  output_count = linear_word(weights, 3u);
  flags = linear_word(activation, 4u);
  weight_input_rows = linear_word(activation, 6u);
  sequence_length = linear_word(activation, 7u);
  if (input_count == 0u || input_count > LINEAR_MAX_INPUTS ||
      sequence_length == 0u || sequence_length > MAMBA_MAX_SEQUENCE ||
      input_count != linear_word(weights, 2u) ||
      output_count != linear_word(activation, 3u) || weight_input_rows > output_count ||
      weight_input_rows > (invocation.element_count - LINEAR_HEADER) / input_count ||
      (flags & ~(LINEAR_FLAG_RESIDUAL | LINEAR_FLAG_RMS_NORM | LINEAR_FLAG_CONVOLUTION)) != 0u ||
      (flags & (LINEAR_FLAG_RMS_NORM | LINEAR_FLAG_CONVOLUTION)) ==
          (LINEAR_FLAG_RMS_NORM | LINEAR_FLAG_CONVOLUTION)) {
    return;
  }
  auxiliary_offset = LINEAR_HEADER + input_count * sequence_length;
  if ((flags & LINEAR_FLAG_RESIDUAL) != 0u) {
    residual_offset = auxiliary_offset;
    if (output_count > (invocation.element_count - auxiliary_offset) / sequence_length) {
      return;
    }
    auxiliary_offset += output_count * sequence_length;
  }
  if ((flags & LINEAR_FLAG_RMS_NORM) != 0u) {
    unsigned time_index;
    rms_weight_offset = auxiliary_offset;
    auxiliary_offset += input_count;
    if (activation[5u] <= 0.0f) {
      return;
    }
    for (time_index = 0u; time_index < sequence_length; ++time_index) {
      float sum_squares = 0.0f;
      unsigned index;
      const float *values = activation + LINEAR_HEADER + time_index * input_count;
      for (index = 0u; index < input_count; ++index) {
        sum_squares += values[index] * values[index];
      }
      rms_scales[time_index] =
          1.0f / sqrtf(sum_squares / (float)input_count + activation[5u]);
    }
  }
  if ((flags & LINEAR_FLAG_CONVOLUTION) != 0u) {
    conv_state_offset = auxiliary_offset;
    conv_weight_offset = conv_state_offset + input_count * LINEAR_CONV_WIDTH;
    conv_bias_offset = conv_weight_offset + input_count * LINEAR_CONV_WIDTH;
    auxiliary_offset = conv_bias_offset + input_count;
  }
  auxiliary_offset = (auxiliary_offset + 3u) & ~3u;
  if (auxiliary_offset > invocation.element_count ||
      output_count > (invocation.element_count - LINEAR_HEADER) / sequence_length ||
      output_count - weight_input_rows >
          (invocation.element_count - auxiliary_offset) / input_count ||
      ((flags & LINEAR_FLAG_CONVOLUTION) != 0u &&
       (output_count > (LINEAR_STATE_OUTPUT_OFFSET - LINEAR_HEADER) / sequence_length ||
        LINEAR_STATE_OUTPUT_OFFSET + input_count * LINEAR_CONV_WIDTH > invocation.element_count ||
        input_count > (invocation.element_count - LINEAR_CONV_OUTPUT_OFFSET) / sequence_length))) {
    return;
  }
  linear_store_word(output, 0u, LINEAR_MAGIC);
  linear_store_word(output, 1u, LINEAR_VERSION);
  linear_store_word(output, 2u, output_count);
  linear_store_word(output, 3u, flags);
  linear_store_word(output, 4u, sequence_length);
  if ((flags & LINEAR_FLAG_CONVOLUTION) != 0u) {
    unsigned index;
    for (index = 0u; index < input_count; ++index) {
      const float *old_state = activation + conv_state_offset + index * LINEAR_CONV_WIDTH;
      float *new_state = output + LINEAR_STATE_OUTPUT_OFFSET + index * LINEAR_CONV_WIDTH;
      float state0 = old_state[0u];
      float state1 = old_state[1u];
      float state2 = old_state[2u];
      float state3 = old_state[3u];
      unsigned time_index;
      for (time_index = 0u; time_index < sequence_length; ++time_index) {
        const float *conv_weight = activation + conv_weight_offset + index * LINEAR_CONV_WIDTH;
        float value;
        state0 = state1;
        state1 = state2;
        state2 = state3;
        state3 = activation[LINEAR_HEADER + time_index * input_count + index];
        value = state0 * conv_weight[0u] + state1 * conv_weight[1u] +
                state2 * conv_weight[2u] + state3 * conv_weight[3u] +
                activation[conv_bias_offset + index];
        output[LINEAR_CONV_OUTPUT_OFFSET + time_index * input_count + index] =
            value / (1.0f + expf(-value));
      }
      new_state[0u] = state0;
      new_state[1u] = state1;
      new_state[2u] = state2;
      new_state[3u] = state3;
    }
  }
  for (output_index = 0u; output_index < output_count; ++output_index) {
    const float *row =
        output_index < weight_input_rows
            ? weights + LINEAR_HEADER + output_index * input_count
            : activation + auxiliary_offset + (output_index - weight_input_rows) * input_count;
    unsigned time_index;
    for (time_index = 0u; time_index + 1u < sequence_length; time_index += 2u) {
      float total0;
      float total1;
      if ((flags & LINEAR_FLAG_RMS_NORM) != 0u) {
        linear_rms_dot_pair(activation + LINEAR_HEADER + time_index * input_count,
                            activation + LINEAR_HEADER + (time_index + 1u) * input_count,
                            activation + rms_weight_offset, row, input_count,
                            rms_scales[time_index], rms_scales[time_index + 1u], &total0, &total1);
      } else if ((flags & LINEAR_FLAG_CONVOLUTION) != 0u) {
        linear_dot_pair(output + LINEAR_CONV_OUTPUT_OFFSET + time_index * input_count,
                        output + LINEAR_CONV_OUTPUT_OFFSET + (time_index + 1u) * input_count, row,
                        input_count, &total0, &total1);
      } else {
        linear_dot_pair(activation + LINEAR_HEADER + time_index * input_count,
                        activation + LINEAR_HEADER + (time_index + 1u) * input_count, row,
                        input_count, &total0, &total1);
      }
      if ((flags & LINEAR_FLAG_RESIDUAL) != 0u) {
        total0 += activation[residual_offset + time_index * output_count + output_index];
        total1 += activation[residual_offset + (time_index + 1u) * output_count + output_index];
      }
      output[LINEAR_HEADER + time_index * output_count + output_index] = total0;
      output[LINEAR_HEADER + (time_index + 1u) * output_count + output_index] = total1;
    }
    if (time_index < sequence_length) {
      float total;
      if ((flags & LINEAR_FLAG_RMS_NORM) != 0u) {
        total = linear_rms_dot(activation + LINEAR_HEADER + time_index * input_count,
                               activation + rms_weight_offset, row, input_count,
                               rms_scales[time_index]);
      } else if ((flags & LINEAR_FLAG_CONVOLUTION) != 0u) {
        total = linear_dot(output + LINEAR_CONV_OUTPUT_OFFSET + time_index * input_count, row,
                           input_count);
      } else {
        total = linear_dot(activation + LINEAR_HEADER + time_index * input_count, row,
                           input_count);
      }
      if ((flags & LINEAR_FLAG_RESIDUAL) != 0u) {
        total += activation[residual_offset + time_index * output_count + output_index];
      }
      output[LINEAR_HEADER + time_index * output_count + output_index] = total;
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
