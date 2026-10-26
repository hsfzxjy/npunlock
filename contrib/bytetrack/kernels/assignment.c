#include <npunlock/npu3720_kernel.h>

#define ASSIGN_CHUNK_ELEMENTS 288u
#define ASSIGN_MAX_CHUNKS_PER_INVOCATION 4u
#define ASSIGN_MAX_ROWS 16u
#define ASSIGN_MAX_COLUMNS 16u
#define ASSIGN_MAX_AUGMENTED_COLUMNS (ASSIGN_MAX_ROWS + ASSIGN_MAX_COLUMNS)
#define ASSIGN_INPUT_MAGIC 372041u
#define ASSIGN_OUTPUT_MAGIC 372042u
#define ASSIGN_FORMAT_VERSION 1u
#define ASSIGN_INPUT_FORBIDDEN_OFFSET 8u
#define ASSIGN_INPUT_COST_OFFSET 32u
#define ASSIGN_OUTPUT_ROW_OFFSET 16u
#define ASSIGN_OUTPUT_COLUMN_OFFSET 32u
#define ASSIGN_OUTPUT_WORDS 50u

#define ASSIGN_STATUS_OK 0u
#define ASSIGN_STATUS_BAD_HEADER 1u
#define ASSIGN_STATUS_BAD_SHAPE 2u
#define ASSIGN_STATUS_BAD_THRESHOLD 3u
#define ASSIGN_STATUS_INTERNAL_BOUND 4u

#define ASSIGN_LARGE_COST 3.0e30f

typedef union assign_float_bits {
  float value;
  unsigned bits;
} assign_float_bits;

static inline int assign_is_finite(float value) {
  assign_float_bits converted;
  converted.value = value;
  return (converted.bits & 0x7f800000u) != 0x7f800000u;
}

static inline float assign_augmented_cost(const float *costs, const unsigned *forbidden,
                                          unsigned columns, float threshold, unsigned row,
                                          unsigned column) {
  if (column < columns) {
    float value = costs[row * ASSIGN_MAX_COLUMNS + column];
    if ((forbidden[row] & (1u << column)) != 0u || value > threshold) {
      return ASSIGN_LARGE_COST;
    }
    return value;
  }
  /* Charging one threshold per unmatched real row differs from the full
     threshold/2 square augmentation only by a constant. */
  return threshold;
}

static inline int assign_hungarian(const float *costs, const unsigned *forbidden, unsigned rows,
                                   unsigned columns, unsigned solve_columns, float threshold,
                                   unsigned *selected_row) {
  float potentials_row[ASSIGN_MAX_ROWS + 1u];
  float potentials_column[ASSIGN_MAX_AUGMENTED_COLUMNS + 1u];
  float minimum[ASSIGN_MAX_AUGMENTED_COLUMNS + 1u];
  unsigned predecessor[ASSIGN_MAX_AUGMENTED_COLUMNS + 1u];
  unsigned used[ASSIGN_MAX_AUGMENTED_COLUMNS + 1u];
  unsigned row;
  unsigned column;

  for (row = 0u; row <= rows; ++row) {
    potentials_row[row] = 0.0f;
  }
  for (column = 0u; column <= solve_columns; ++column) {
    potentials_column[column] = 0.0f;
    selected_row[column] = 0u;
    used[column] = 0u;
  }
  for (row = 1u; row <= rows; ++row) {
    unsigned current_column = 0u;
    selected_row[0] = row;
    for (column = 0u; column <= solve_columns; ++column) {
      minimum[column] = ASSIGN_LARGE_COST;
      predecessor[column] = 0u;
    }
    do {
      unsigned current_row;
      unsigned next_column = 0u;
      float delta = ASSIGN_LARGE_COST;
      used[current_column] = row;
      current_row = selected_row[current_column];
      for (column = 1u; column <= solve_columns; ++column) {
        if (used[column] != row) {
          float reduced = assign_augmented_cost(costs, forbidden, columns, threshold,
                                                current_row - 1u, column - 1u) -
                          potentials_row[current_row] - potentials_column[column];
          if (reduced < minimum[column] ||
              (reduced == minimum[column] && current_column < predecessor[column])) {
            minimum[column] = reduced;
            predecessor[column] = current_column;
          }
          if (minimum[column] < delta ||
              (minimum[column] == delta && (next_column == 0u || column < next_column))) {
            delta = minimum[column];
            next_column = column;
          }
        }
      }
      if (next_column == 0u || delta >= ASSIGN_LARGE_COST) {
        return 0;
      }
      for (column = 0u; column <= solve_columns; ++column) {
        if (used[column] == row) {
          potentials_row[selected_row[column]] += delta;
          potentials_column[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      current_column = next_column;
    } while (selected_row[current_column] != 0u);

    do {
      unsigned previous_column = predecessor[current_column];
      selected_row[current_column] = selected_row[previous_column];
      current_column = previous_column;
    } while (current_column != 0u);
  }
  return 1;
}

static void assign_clear_output(float *output) {
  unsigned index;
  for (index = 0u; index < ASSIGN_OUTPUT_WORDS; ++index) {
    output[index] = 0.0f;
  }
}

static void assign_write_failure(float *output, unsigned status, unsigned rows, unsigned columns) {
  unsigned *words = (unsigned *)output;
  assign_clear_output(output);
  words[0] = ASSIGN_OUTPUT_MAGIC;
  words[1] = ASSIGN_FORMAT_VERSION;
  words[2] = status;
  words[3] = rows;
  words[4] = columns;
}

static void assign_solve_chunk(const float *input, float *output) {
  const unsigned *input_words = (const unsigned *)input;
  const unsigned *forbidden = input_words + ASSIGN_INPUT_FORBIDDEN_OFFSET;
  const float *costs = input + ASSIGN_INPUT_COST_OFFSET;
  unsigned *output_words = (unsigned *)output;
  unsigned selected_row[ASSIGN_MAX_AUGMENTED_COLUMNS + 1u];
  int row_to_column[ASSIGN_MAX_ROWS];
  int column_to_row[ASSIGN_MAX_COLUMNS];
  unsigned rows;
  unsigned columns;
  unsigned augmented_columns;
  unsigned row;
  unsigned column;
  unsigned match_count = 0u;
  int solved = 0;
  float threshold;
  float match_cost = 0.0f;

  assign_clear_output(output);
  if (input_words[0] != ASSIGN_INPUT_MAGIC || input_words[1] != ASSIGN_FORMAT_VERSION ||
      input_words[5] != 0u || input_words[6] != ASSIGN_MAX_ROWS ||
      input_words[7] != ASSIGN_MAX_COLUMNS) {
    assign_write_failure(output, ASSIGN_STATUS_BAD_HEADER, 0u, 0u);
    return;
  }
  rows = input_words[2];
  columns = input_words[3];
  threshold = input[4];
  if (rows > ASSIGN_MAX_ROWS || columns > ASSIGN_MAX_COLUMNS) {
    assign_write_failure(output, ASSIGN_STATUS_BAD_SHAPE, rows, columns);
    return;
  }
  if (!assign_is_finite(threshold) || threshold < 0.0f) {
    assign_write_failure(output, ASSIGN_STATUS_BAD_THRESHOLD, rows, columns);
    return;
  }
  augmented_columns = rows + columns;
  if (augmented_columns > ASSIGN_MAX_AUGMENTED_COLUMNS) {
    assign_write_failure(output, ASSIGN_STATUS_INTERNAL_BOUND, rows, columns);
    return;
  }
  for (row = 0u; row < rows; ++row) {
    row_to_column[row] = -1;
  }
  for (column = 0u; column < columns; ++column) {
    column_to_row[column] = -1;
  }

  solved =
      assign_hungarian(costs, forbidden, rows, columns, augmented_columns, threshold, selected_row);
  if (!solved) {
    assign_write_failure(output, ASSIGN_STATUS_INTERNAL_BOUND, rows, columns);
    return;
  }

  for (column = 1u; column <= columns; ++column) {
    unsigned assigned_row = selected_row[column];
    if (assigned_row != 0u && assigned_row <= rows) {
      unsigned real_row = assigned_row - 1u;
      unsigned real_column = column - 1u;
      float value = costs[real_row * ASSIGN_MAX_COLUMNS + real_column];
      if ((forbidden[real_row] & (1u << real_column)) == 0u && value <= threshold) {
        row_to_column[real_row] = (int)real_column;
        column_to_row[real_column] = (int)real_row;
        match_cost += value;
        ++match_count;
      }
    }
  }

  output_words[0] = ASSIGN_OUTPUT_MAGIC;
  output_words[1] = ASSIGN_FORMAT_VERSION;
  output_words[2] = ASSIGN_STATUS_OK;
  output_words[3] = rows;
  output_words[4] = columns;
  output_words[5] = match_count;
  output_words[6] = rows - match_count;
  output_words[7] = columns - match_count;
  output_words[8] = 0u;
  output_words[9] = augmented_columns;
  output_words[10] = 0u;
  for (row = 0u; row < ASSIGN_MAX_ROWS; ++row) {
    output_words[ASSIGN_OUTPUT_ROW_OFFSET + row] = row < rows ? (unsigned)row_to_column[row] : ~0u;
  }
  for (column = 0u; column < ASSIGN_MAX_COLUMNS; ++column) {
    output_words[ASSIGN_OUTPUT_COLUMN_OFFSET + column] =
        column < columns ? (unsigned)column_to_row[column] : ~0u;
  }
  output[48] = match_cost;
  output[49] = match_cost + (float)(rows + columns - 2u * match_count) * threshold * 0.5f;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *input;
  float *output;
  unsigned chunk_count;
  unsigned chunk;

  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  if (invocation.element_count == 0u ||
      invocation.element_count > ASSIGN_CHUNK_ELEMENTS * ASSIGN_MAX_CHUNKS_PER_INVOCATION ||
      invocation.element_count % ASSIGN_CHUNK_ELEMENTS != 0u) {
    return;
  }
  input = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  chunk_count = invocation.element_count / ASSIGN_CHUNK_ELEMENTS;
  for (chunk = 0u; chunk < chunk_count; ++chunk) {
    assign_solve_chunk(input + chunk * ASSIGN_CHUNK_ELEMENTS,
                       output + chunk * ASSIGN_CHUNK_ELEMENTS);
  }
}
