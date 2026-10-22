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
  unsigned char processed[NMS_CAPACITY];
  unsigned char suppressed[NMS_CAPACITY];
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
  for (unsigned i = 0u; i < NMS_CAPACITY; ++i) {
    processed[i] = 0u;
    suppressed[i] = 0u;
  }

  output[1] = input[1];
  output[2] = input[2];
  output[3] = input[3];
  output[4] = input[0];
  output[5] = (__fp16)1.0f;

  while (selected_count < max_detections) {
    unsigned best = NMS_CAPACITY;
    unsigned best_original_index = 0u;
    float best_score = -1.0f;

    for (unsigned i = 0u; i < candidate_count; ++i) {
      const __fp16 *record = input + NMS_HEADER_WORDS + i * NMS_RECORD_WORDS;
      float score = (float)record[4];
      unsigned original_index;
      if (processed[i] != 0u || suppressed[i] != 0u || !(score > confidence_threshold)) {
        continue;
      }
      original_index = nms_original_index(record);
      if (best == NMS_CAPACITY || score > best_score ||
          (score == best_score && original_index < best_original_index)) {
        best = i;
        best_score = score;
        best_original_index = original_index;
      }
    }
    if (best == NMS_CAPACITY) {
      break;
    }

    processed[best] = 1u;
    {
      const __fp16 *best_record = input + NMS_HEADER_WORDS + best * NMS_RECORD_WORDS;
      __fp16 *output_record = output + NMS_HEADER_WORDS + selected_count * NMS_RECORD_WORDS;
      unsigned best_class = (unsigned)(float)best_record[5];
      for (unsigned field = 0u; field < NMS_RECORD_WORDS; ++field) {
        output_record[field] = best_record[field];
      }
      ++selected_count;

      for (unsigned candidate = 0u; candidate < candidate_count; ++candidate) {
        const __fp16 *record;
        unsigned candidate_class;
        if (processed[candidate] != 0u || suppressed[candidate] != 0u) {
          continue;
        }
        record = input + NMS_HEADER_WORDS + candidate * NMS_RECORD_WORDS;
        candidate_class = (unsigned)(float)record[5];
        if (candidate_class == best_class && nms_iou(best_record, record) > iou_threshold) {
          suppressed[candidate] = 1u;
        }
      }
    }
  }
  output[0] = (__fp16)(float)selected_count;
}
