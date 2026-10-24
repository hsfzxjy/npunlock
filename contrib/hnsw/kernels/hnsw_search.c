#include <npunlock/npu3720_kernel.h>

#define HNSW_CHUNK_ELEMENTS 1024u
#define HNSW_MAX_CHUNKS_PER_INVOCATION 80u
#define HNSW_MAX_NODES 48u
#define HNSW_MAX_DIMENSIONS 32u
#define HNSW_MAX_LEVELS 4u
#define HNSW_MAX_M 4u
#define HNSW_MAX_EF 24u
#define HNSW_MAX_K 16u
#define HNSW_QUERY_MAGIC 372021u
#define HNSW_GRAPH_MAGIC 372022u
#define HNSW_OUTPUT_MAGIC 372023u
#define HNSW_FORMAT_VERSION 4u
#define HNSW_QUERY_HEADER 8u
#define HNSW_GRAPH_HEADER 8u
#define HNSW_OUTPUT_HEADER 16u
#define HNSW_OUTPUT_ELEMENTS (HNSW_OUTPUT_HEADER + HNSW_MAX_K * 2u)
#define HNSW_NODE_GRAPH_ELEMENTS (1u + HNSW_MAX_LEVELS * (1u + HNSW_MAX_M))
#define HNSW_VISITED_WORDS ((HNSW_MAX_NODES + 31u) / 32u)

#define HNSW_STATUS_OK 0u
#define HNSW_STATUS_BAD_QUERY 1u
#define HNSW_STATUS_BAD_GRAPH 2u
#define HNSW_STATUS_BAD_SEARCH 3u
#define HNSW_STATUS_INTERNAL_BOUND 4u

typedef float hnsw_float4 __attribute__((ext_vector_type(4)));

static int hnsw_better(float left_distance, unsigned left_node, float right_distance,
                       unsigned right_node) {
  return left_distance < right_distance ||
         (left_distance == right_distance && left_node < right_node);
}

static float hnsw_distance(const float *query, const float *vectors, unsigned dimensions,
                           unsigned node) {
  float total = 0.0f;
  const float *candidate = vectors + node * dimensions;
  unsigned index;
  if (dimensions == 16u) {
    /* Fixed four-lane reduction tree mirrored by the host oracle. */
    const hnsw_float4 *query4 = (const hnsw_float4 *)query;
    const hnsw_float4 *candidate4 = (const hnsw_float4 *)candidate;
    hnsw_float4 delta0 = query4[0] - candidate4[0];
    hnsw_float4 delta1 = query4[1] - candidate4[1];
    hnsw_float4 delta2 = query4[2] - candidate4[2];
    hnsw_float4 delta3 = query4[3] - candidate4[3];
    hnsw_float4 squared0 = delta0 * delta0;
    hnsw_float4 squared1 = delta1 * delta1;
    hnsw_float4 squared2 = delta2 * delta2;
    hnsw_float4 squared3 = delta3 * delta3;
    hnsw_float4 sums = (squared0 + squared1) + (squared2 + squared3);
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
  }
  for (index = 0u; index < dimensions; ++index) {
    float delta = query[index] - candidate[index];
    total += delta * delta;
  }
  return total;
}

static unsigned hnsw_graph_record(unsigned node) {
  return HNSW_GRAPH_HEADER + node * HNSW_NODE_GRAPH_ELEMENTS;
}

static unsigned hnsw_graph_layer(unsigned node, unsigned layer) {
  return hnsw_graph_record(node) + 1u + layer * (1u + HNSW_MAX_M);
}

static void hnsw_candidate_push(float *distances, unsigned *nodes, unsigned *size, float distance,
                                unsigned node) {
  unsigned position = *size;
  ++*size;
  while (position != 0u) {
    unsigned parent = (position - 1u) / 2u;
    if (!hnsw_better(distance, node, distances[parent], nodes[parent])) {
      break;
    }
    distances[position] = distances[parent];
    nodes[position] = nodes[parent];
    position = parent;
  }
  distances[position] = distance;
  nodes[position] = node;
}

static void hnsw_candidate_pop(float *distances, unsigned *nodes, unsigned *size, float *distance,
                               unsigned *node) {
  unsigned old_size = *size;
  float replacement_distance = distances[old_size - 1u];
  unsigned replacement_node = nodes[old_size - 1u];
  unsigned position = 0u;
  *distance = distances[0];
  *node = nodes[0];
  --*size;
  while (position * 2u + 1u < *size) {
    unsigned child = position * 2u + 1u;
    if (child + 1u < *size &&
        hnsw_better(distances[child + 1u], nodes[child + 1u], distances[child], nodes[child])) {
      ++child;
    }
    if (!hnsw_better(distances[child], nodes[child], replacement_distance, replacement_node)) {
      break;
    }
    distances[position] = distances[child];
    nodes[position] = nodes[child];
    position = child;
  }
  if (*size != 0u) {
    distances[position] = replacement_distance;
    nodes[position] = replacement_node;
  }
}

static int hnsw_worse(float left_distance, unsigned left_node, float right_distance,
                      unsigned right_node) {
  return left_distance > right_distance ||
         (left_distance == right_distance && left_node > right_node);
}

static void hnsw_top_push(float *distances, unsigned *nodes, unsigned *size, float distance,
                          unsigned node) {
  unsigned position = *size;
  ++*size;
  while (position != 0u) {
    unsigned parent = (position - 1u) / 2u;
    if (!hnsw_worse(distance, node, distances[parent], nodes[parent])) {
      break;
    }
    distances[position] = distances[parent];
    nodes[position] = nodes[parent];
    position = parent;
  }
  distances[position] = distance;
  nodes[position] = node;
}

static void hnsw_top_pop(float *distances, unsigned *nodes, unsigned *size) {
  unsigned old_size = *size;
  float replacement_distance = distances[old_size - 1u];
  unsigned replacement_node = nodes[old_size - 1u];
  unsigned position = 0u;
  --*size;
  while (position * 2u + 1u < *size) {
    unsigned child = position * 2u + 1u;
    if (child + 1u < *size &&
        hnsw_worse(distances[child + 1u], nodes[child + 1u], distances[child], nodes[child])) {
      ++child;
    }
    if (!hnsw_worse(distances[child], nodes[child], replacement_distance, replacement_node)) {
      break;
    }
    distances[position] = distances[child];
    nodes[position] = nodes[child];
    position = child;
  }
  if (*size != 0u) {
    distances[position] = replacement_distance;
    nodes[position] = replacement_node;
  }
}

static void hnsw_top_replace(float *distances, unsigned *nodes, unsigned size, float distance,
                             unsigned node) {
  unsigned position = 0u;
  while (position * 2u + 1u < size) {
    unsigned child = position * 2u + 1u;
    if (child + 1u < size &&
        hnsw_worse(distances[child + 1u], nodes[child + 1u], distances[child], nodes[child])) {
      ++child;
    }
    if (!hnsw_worse(distances[child], nodes[child], distance, node)) {
      break;
    }
    distances[position] = distances[child];
    nodes[position] = nodes[child];
    position = child;
  }
  distances[position] = distance;
  nodes[position] = node;
}

static void hnsw_write_status(unsigned *output, unsigned status) {
  output[0] = HNSW_OUTPUT_MAGIC;
  output[1] = status;
}

static void __attribute__((noinline)) hnsw_write_failure(float *output, unsigned status) {
  unsigned value;
  for (value = 0u; value < HNSW_OUTPUT_ELEMENTS; ++value) {
    output[value] = 0.0f;
  }
  hnsw_write_status((unsigned *)output, status);
}

static unsigned __attribute__((noinline)) hnsw_validate_graph(const float *graph) {
  const unsigned *graph_words = (const unsigned *)graph;
  unsigned node_count = graph_words[2];
  unsigned dimensions = graph_words[3];
  unsigned entry_point = graph_words[4];
  unsigned max_level = graph_words[5];
  unsigned node;
  unsigned layer;
  if (graph_words[0] != HNSW_GRAPH_MAGIC || graph_words[1] != HNSW_FORMAT_VERSION ||
      node_count == 0u || node_count > HNSW_MAX_NODES || dimensions == 0u ||
      dimensions > HNSW_MAX_DIMENSIONS || entry_point >= node_count ||
      max_level >= HNSW_MAX_LEVELS || graph_words[6] != HNSW_MAX_LEVELS ||
      graph_words[7] == 0u || graph_words[7] > HNSW_MAX_M ||
      HNSW_GRAPH_HEADER + node_count * HNSW_NODE_GRAPH_ELEMENTS > HNSW_CHUNK_ELEMENTS) {
    return HNSW_STATUS_BAD_GRAPH;
  }
  if (graph_words[hnsw_graph_record(entry_point)] != max_level) {
    return HNSW_STATUS_BAD_GRAPH;
  }
  for (node = 0u; node < node_count; ++node) {
    unsigned node_level = graph_words[hnsw_graph_record(node)];
    if (node_level >= HNSW_MAX_LEVELS || node_level > max_level) {
      return HNSW_STATUS_BAD_GRAPH;
    }
    for (layer = 0u; layer < HNSW_MAX_LEVELS; ++layer) {
      unsigned list = hnsw_graph_layer(node, layer);
      unsigned neighbor_count = graph_words[list];
      unsigned slot;
      if (neighbor_count > HNSW_MAX_M || (layer > node_level && neighbor_count != 0u)) {
        return HNSW_STATUS_BAD_GRAPH;
      }
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor = graph_words[list + 1u + slot];
        if (neighbor >= node_count || neighbor == node ||
            graph_words[hnsw_graph_record(neighbor)] < layer) {
          return HNSW_STATUS_BAD_GRAPH;
        }
      }
    }
  }
  return HNSW_STATUS_OK;
}

static void hnsw_search_chunk(const float *query_payload, const unsigned *query_words,
                              const float *graph, float *output) {
  const unsigned *graph_words = (const unsigned *)graph;
  unsigned *output_words = (unsigned *)output;
  const float *query;
  const float *vectors;
  volatile unsigned visited[HNSW_VISITED_WORDS];
  float candidate_distances[HNSW_MAX_NODES];
  unsigned candidate_nodes[HNSW_MAX_NODES];
  float top_distances[HNSW_MAX_EF];
  unsigned top_nodes[HNSW_MAX_EF];
  unsigned node_count;
  unsigned dimensions;
  unsigned entry_point;
  unsigned max_level;
  unsigned ef_search;
  unsigned result_count;
  unsigned value;
  unsigned node;
  unsigned layer;
  unsigned slot;
  unsigned current;
  float current_distance;
  unsigned distance_count = 0u;
  unsigned upper_hops = 0u;
  unsigned base_expansions = 0u;
  unsigned terminated_by_bound = 0u;
  unsigned candidate_size = 0u;
  unsigned candidate_peak = 0u;
  unsigned top_size = 0u;
  unsigned top_peak = 0u;
  unsigned visited_count = 0u;

  node_count = query_words[2];
  dimensions = query_words[3];
  entry_point = query_words[4];
  max_level = query_words[5];
  ef_search = query_words[6];
  result_count = query_words[7];
  query = query_payload + HNSW_QUERY_HEADER;
  vectors = query + dimensions;

  current = entry_point;
  current_distance = hnsw_distance(query, vectors, dimensions, current);
  distance_count = 1u;
  for (layer = max_level; layer > 0u; --layer) {
    unsigned step;
    for (step = 0u; step < node_count; ++step) {
      unsigned list = hnsw_graph_layer(current, layer);
      unsigned neighbor_count;
      unsigned best = current;
      float best_distance = current_distance;
      neighbor_count = graph_words[list];
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor = graph_words[list + 1u + slot];
        float distance;
        distance = hnsw_distance(query, vectors, dimensions, neighbor);
        ++distance_count;
        if (hnsw_better(distance, neighbor, best_distance, best)) {
          best = neighbor;
          best_distance = distance;
        }
      }
      if (best == current) {
        break;
      }
      current = best;
      current_distance = best_distance;
      ++upper_hops;
    }
    if (step == node_count) {
      hnsw_write_failure(output, HNSW_STATUS_INTERNAL_BOUND);
      return;
    }
  }

  for (node = 0u; node < HNSW_VISITED_WORDS; ++node) {
    visited[node] = 0u;
  }
  visited[current >> 5u] |= 1u << (current & 31u);
  visited_count = 1u;
  hnsw_candidate_push(candidate_distances, candidate_nodes, &candidate_size, current_distance,
                      current);
  hnsw_top_push(top_distances, top_nodes, &top_size, current_distance, current);
  candidate_peak = 1u;
  top_peak = 1u;

  while (candidate_size != 0u) {
    float candidate_distance;
    unsigned candidate;
    hnsw_candidate_pop(candidate_distances, candidate_nodes, &candidate_size, &candidate_distance,
                       &candidate);
    if (top_size >= ef_search &&
        hnsw_worse(candidate_distance, candidate, top_distances[0], top_nodes[0])) {
      terminated_by_bound = 1u;
      break;
    }
    ++base_expansions;
    {
      unsigned list = hnsw_graph_layer(candidate, 0u);
      unsigned neighbor_count = graph_words[list];
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor = graph_words[list + 1u + slot];
        unsigned visited_word = neighbor >> 5u;
        unsigned visited_mask = 1u << (neighbor & 31u);
        if ((visited[visited_word] & visited_mask) == 0u) {
          float distance;
          visited[visited_word] |= visited_mask;
          ++visited_count;
          distance = hnsw_distance(query, vectors, dimensions, neighbor);
          ++distance_count;
          if (top_size < ef_search ||
              hnsw_better(distance, neighbor, top_distances[0], top_nodes[0])) {
            if (candidate_size >= HNSW_MAX_NODES) {
              hnsw_write_failure(output, HNSW_STATUS_INTERNAL_BOUND);
              return;
            }
            hnsw_candidate_push(candidate_distances, candidate_nodes, &candidate_size, distance,
                                neighbor);
            if (top_size < ef_search) {
              hnsw_top_push(top_distances, top_nodes, &top_size, distance, neighbor);
            } else {
              hnsw_top_replace(top_distances, top_nodes, top_size, distance, neighbor);
            }
            if (candidate_size > candidate_peak) {
              candidate_peak = candidate_size;
            }
            if (top_size > top_peak) {
              top_peak = top_size;
            }
          }
        }
      }
    }
  }

  if (top_size < result_count) {
    hnsw_write_failure(output, HNSW_STATUS_BAD_SEARCH);
    return;
  }
  for (value = HNSW_OUTPUT_HEADER + result_count * 2u; value < HNSW_OUTPUT_ELEMENTS; ++value) {
    output[value] = 0.0f;
  }
  output_words[0] = HNSW_OUTPUT_MAGIC;
  output_words[1] = HNSW_STATUS_OK;
  output_words[2] = node_count;
  output_words[3] = dimensions;
  output_words[4] = entry_point;
  output_words[5] = max_level;
  output_words[6] = ef_search;
  output_words[7] = result_count;
  output_words[8] = visited_count;
  output_words[9] = distance_count;
  output_words[10] = upper_hops;
  output_words[11] = base_expansions;
  output_words[12] = terminated_by_bound;
  output_words[13] = candidate_peak;
  output_words[14] = top_peak;
  output_words[15] = result_count;

  while (top_size > result_count) {
    hnsw_top_pop(top_distances, top_nodes, &top_size);
  }
  for (value = result_count; value > 0u; --value) {
    unsigned output_index = HNSW_OUTPUT_HEADER + (value - 1u) * 2u;
    output_words[output_index] = top_nodes[0];
    output[output_index + 1u] = top_distances[0];
    hnsw_top_pop(top_distances, top_nodes, &top_size);
  }
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *query_payload;
  const float *graph;
  float *output;
  unsigned chunk_count;
  unsigned chunk;
  unsigned graph_status;
  const unsigned *query_words;

  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  if (invocation.element_count == 0u ||
      invocation.element_count > HNSW_CHUNK_ELEMENTS * HNSW_MAX_CHUNKS_PER_INVOCATION ||
      invocation.element_count % HNSW_CHUNK_ELEMENTS != 0u) {
    return;
  }
  query_payload = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  graph = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  chunk_count = invocation.element_count / HNSW_CHUNK_ELEMENTS;
  query_words = (const unsigned *)query_payload;
  graph_status = hnsw_validate_graph(graph);
  if (graph_status != HNSW_STATUS_OK) {
    for (chunk = 0u; chunk < chunk_count; ++chunk) {
      hnsw_write_failure(output + chunk * HNSW_CHUNK_ELEMENTS, graph_status);
    }
    return;
  }
  if (query_words[0] != HNSW_QUERY_MAGIC || query_words[1] != HNSW_FORMAT_VERSION ||
      query_words[2] != ((const unsigned *)graph)[2] ||
      query_words[3] != ((const unsigned *)graph)[3] ||
      query_words[4] != ((const unsigned *)graph)[4] ||
      query_words[5] != ((const unsigned *)graph)[5] || query_words[6] == 0u ||
      query_words[6] > HNSW_MAX_EF || query_words[6] > query_words[2] ||
      query_words[7] == 0u || query_words[7] > HNSW_MAX_K ||
      query_words[7] > query_words[6] ||
      HNSW_QUERY_HEADER + query_words[3] * (query_words[2] + 1u) > HNSW_CHUNK_ELEMENTS) {
    for (chunk = 0u; chunk < chunk_count; ++chunk) {
      hnsw_write_failure(output + chunk * HNSW_CHUNK_ELEMENTS, HNSW_STATUS_BAD_QUERY);
    }
    return;
  }
  for (chunk = 0u; chunk < chunk_count; ++chunk) {
    unsigned offset = chunk * HNSW_CHUNK_ELEMENTS;
    hnsw_search_chunk(query_payload + offset, query_words, graph, output + offset);
  }
}
