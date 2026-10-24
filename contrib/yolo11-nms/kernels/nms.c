#include <npunlock/npu3720_kernel.h>

#define NMS_CHUNK_WORDS 2048u
#define NMS_HEADER_WORDS 8u
#define NMS_RECORD_WORDS 8u
#define NMS_CAPACITY 255u

static float nms_maxf(float left, float right) { return left > right ? left : right; }

static float nms_minf(float left, float right) { return left < right ? left : right; }

static unsigned nms_original_index(const __fp16 *record) {
  unsigned low = (unsigned)(float)record[6];
  unsigned high = (unsigned)(float)record[7];
  return low + high * 256u;
}

static float nms_iou(const __fp16 *left, const __fp16 *right);

static int nms_is_suppressed(const __fp16 *candidate, const __fp16 *output, unsigned selected_count,
                             float iou_threshold) {
  unsigned candidate_class = (unsigned)(float)candidate[5];
  for (unsigned selected = 0u; selected < selected_count; ++selected) {
    const __fp16 *record = output + NMS_HEADER_WORDS + selected * NMS_RECORD_WORDS;
    if ((unsigned)(float)record[5] == candidate_class &&
        nms_iou(record, candidate) > iou_threshold) {
      return 1;
    }
  }
  return 0;
}

static void nms_append(const __fp16 *candidate, __fp16 *output, unsigned selected_count) {
  __fp16 *record = output + NMS_HEADER_WORDS + selected_count * NMS_RECORD_WORDS;
  for (unsigned field = 0u; field < NMS_RECORD_WORDS; ++field) {
    record[field] = candidate[field];
  }
}

static float nms_iou(const __fp16 *left, const __fp16 *right) {
  float left_x1 = (float)left[0];
  float left_y1 = (float)left[1];
  float left_x2 = (float)left[2];
  float left_y2 = (float)left[3];
  float right_x1 = (float)right[0];
  float right_y1 = (float)right[1];
  float right_x2 = (float)right[2];
  float right_y2 = (float)right[3];
  float intersection_width = nms_minf(left_x2, right_x2) - nms_maxf(left_x1, right_x1);
  float intersection_height = nms_minf(left_y2, right_y2) - nms_maxf(left_y1, right_y1);
  float left_width = left_x2 - left_x1;
  float left_height = left_y2 - left_y1;
  float right_width = right_x2 - right_x1;
  float right_height = right_y2 - right_y1;
  float intersection;
  float union_area;

  if (intersection_width <= 0.0f || intersection_height <= 0.0f || left_width <= 0.0f ||
      left_height <= 0.0f || right_width <= 0.0f || right_height <= 0.0f) {
    return 0.0f;
  }
  intersection = intersection_width * intersection_height;
  union_area = left_width * left_height + right_width * right_height - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const __fp16 *input;
  __fp16 *output;
  unsigned candidate_count;
  unsigned max_detections;
  unsigned selected_count = 0u;
  float candidate_count_value;
  float confidence_threshold;
  float iou_threshold;
  float max_detections_value;

  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  if (invocation.element_count != NMS_CHUNK_WORDS) {
    return;
  }
  input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
  output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
  candidate_count_value = (float)input[0];
  confidence_threshold = (float)input[1];
  iou_threshold = (float)input[2];
  max_detections_value = (float)input[3];
  if (!(candidate_count_value >= 0.0f && candidate_count_value <= (float)NMS_CAPACITY) ||
      !(max_detections_value >= 0.0f && max_detections_value <= (float)NMS_CAPACITY) ||
      !(confidence_threshold >= 0.0f && confidence_threshold <= 1.0f) ||
      !(iou_threshold >= 0.0f && iou_threshold <= 1.0f)) {
    return;
  }
  candidate_count = (unsigned)candidate_count_value;
  max_detections = (unsigned)max_detections_value;
  if ((float)candidate_count != candidate_count_value ||
      (float)max_detections != max_detections_value) {
    return;
  }

  for (unsigned i = 0u; i < NMS_CHUNK_WORDS; ++i) {
    output[i] = (__fp16)0.0f;
  }
  output[1] = input[1];
  output[2] = input[2];
  output[3] = input[3];
  output[4] = input[0];
  output[5] = (__fp16)2.0f;

  /* The preceding stable NPU TopK supplies non-increasing scores. Process
   * equal FP16-score groups by original index so transport quantization does
   * not change the deterministic reference order. */
  for (unsigned group_begin = 0u;
       group_begin < candidate_count && selected_count < max_detections;) {
    const __fp16 *first = input + NMS_HEADER_WORDS + group_begin * NMS_RECORD_WORDS;
    float score = (float)first[4];
    unsigned group_end = group_begin + 1u;
    unsigned last_original = 0u;
    unsigned last_position = 0u;
    int have_last = 0;

    if (!(score > confidence_threshold)) {
      break;
    }
    while (group_end < candidate_count) {
      const __fp16 *next = input + NMS_HEADER_WORDS + group_end * NMS_RECORD_WORDS;
      float next_score = (float)next[4];
      if (next_score > score) {
        output[6] = (__fp16)1.0f;
        output[0] = (__fp16)0.0f;
        return;
      }
      if (next_score != score) {
        break;
      }
      ++group_end;
    }

    for (unsigned emitted = group_begin; emitted < group_end && selected_count < max_detections;
         ++emitted) {
      unsigned best_position = NMS_CAPACITY;
      unsigned best_original = 0u;
      for (unsigned position = group_begin; position < group_end; ++position) {
        const __fp16 *candidate = input + NMS_HEADER_WORDS + position * NMS_RECORD_WORDS;
        unsigned original = nms_original_index(candidate);
        if (have_last != 0 && (original < last_original ||
                               (original == last_original && position <= last_position))) {
          continue;
        }
        if (best_position == NMS_CAPACITY || original < best_original ||
            (original == best_original && position < best_position)) {
          best_position = position;
          best_original = original;
        }
      }
      if (best_position == NMS_CAPACITY) {
        output[6] = (__fp16)2.0f;
        output[0] = (__fp16)0.0f;
        return;
      }
      {
        const __fp16 *candidate = input + NMS_HEADER_WORDS + best_position * NMS_RECORD_WORDS;
        if (!nms_is_suppressed(candidate, output, selected_count, iou_threshold)) {
          nms_append(candidate, output, selected_count);
          ++selected_count;
        }
      }
      last_original = best_original;
      last_position = best_position;
      have_last = 1;
    }
    group_begin = group_end;
  }
  output[0] = (__fp16)(float)selected_count;
}
