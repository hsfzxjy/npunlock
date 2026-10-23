#include <npunlock/npu3720_kernel.h>

#define HNSW_CHUNK_ELEMENTS 1024u
#define HNSW_MAX_NODES 48u
#define HNSW_MAX_DIMENSIONS 32u
#define HNSW_MAX_LEVELS 4u
#define HNSW_MAX_M 4u
#define HNSW_MAX_EF 24u
#define HNSW_MAX_K 16u
#define HNSW_QUERY_MAGIC 372021u
#define HNSW_GRAPH_MAGIC 372022u
#define HNSW_OUTPUT_MAGIC 372023u
#define HNSW_FORMAT_VERSION 1u
#define HNSW_QUERY_HEADER 8u
#define HNSW_GRAPH_HEADER 8u
#define HNSW_OUTPUT_HEADER 16u
#define HNSW_NODE_GRAPH_ELEMENTS (1u + HNSW_MAX_LEVELS * (1u + HNSW_MAX_M))

#define HNSW_STATUS_OK 0u
#define HNSW_STATUS_BAD_QUERY 1u
#define HNSW_STATUS_BAD_GRAPH 2u
#define HNSW_STATUS_BAD_SEARCH 3u
#define HNSW_STATUS_INTERNAL_BOUND 4u

static int hnsw_exact_uint(float value, unsigned maximum, unsigned *result) {
  unsigned converted;
  if (!(value >= 0.0f) || value > (float)maximum) {
    return 0;
  }
  converted = (unsigned)value;
  if ((float)converted != value) {
    return 0;
  }
  *result = converted;
  return 1;
}

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

static void hnsw_write_status(float *output, unsigned status) {
  output[0] = (float)HNSW_OUTPUT_MAGIC;
  output[1] = (float)status;
}

void controlled_act(unsigned layerParams) {
  act_abi_invocation invocation;
  const float *query_payload;
  const float *graph;
  float *output;
  const float *query;
  const float *vectors;
  volatile unsigned char visited[HNSW_MAX_NODES];
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

  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
  if (invocation.element_count != HNSW_CHUNK_ELEMENTS) {
    return;
  }
  query_payload = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
  graph = ACT_ABI_INPUT_PTR32(const float, invocation, 1u);
  output = ACT_ABI_OUTPUT_PTR32(float, invocation, 2u);
  for (value = 0u; value < HNSW_CHUNK_ELEMENTS; ++value) {
    output[value] = 0.0f;
  }
  hnsw_write_status(output, HNSW_STATUS_BAD_QUERY);
  if (!hnsw_exact_uint(query_payload[0], HNSW_QUERY_MAGIC, &value) || value != HNSW_QUERY_MAGIC ||
      !hnsw_exact_uint(query_payload[1], HNSW_FORMAT_VERSION, &value) ||
      value != HNSW_FORMAT_VERSION ||
      !hnsw_exact_uint(query_payload[2], HNSW_MAX_NODES, &node_count) || node_count == 0u ||
      !hnsw_exact_uint(query_payload[3], HNSW_MAX_DIMENSIONS, &dimensions) || dimensions == 0u ||
      !hnsw_exact_uint(query_payload[4], node_count - 1u, &entry_point) ||
      !hnsw_exact_uint(query_payload[5], HNSW_MAX_LEVELS - 1u, &max_level) ||
      !hnsw_exact_uint(query_payload[6], HNSW_MAX_EF, &ef_search) || ef_search == 0u ||
      ef_search > node_count || !hnsw_exact_uint(query_payload[7], HNSW_MAX_K, &result_count) ||
      result_count == 0u || result_count > ef_search ||
      HNSW_QUERY_HEADER + dimensions * (node_count + 1u) > HNSW_CHUNK_ELEMENTS) {
    return;
  }
  query = query_payload + HNSW_QUERY_HEADER;
  vectors = query + dimensions;

  hnsw_write_status(output, HNSW_STATUS_BAD_GRAPH);
  if (!hnsw_exact_uint(graph[0], HNSW_GRAPH_MAGIC, &value) || value != HNSW_GRAPH_MAGIC ||
      !hnsw_exact_uint(graph[1], HNSW_FORMAT_VERSION, &value) || value != HNSW_FORMAT_VERSION ||
      !hnsw_exact_uint(graph[2], HNSW_MAX_NODES, &value) || value != node_count ||
      !hnsw_exact_uint(graph[3], HNSW_MAX_DIMENSIONS, &value) || value != dimensions ||
      !hnsw_exact_uint(graph[4], node_count - 1u, &value) || value != entry_point ||
      !hnsw_exact_uint(graph[5], HNSW_MAX_LEVELS - 1u, &value) || value != max_level ||
      !hnsw_exact_uint(graph[6], HNSW_MAX_LEVELS, &value) || value != HNSW_MAX_LEVELS ||
      !hnsw_exact_uint(graph[7], HNSW_MAX_M, &value) || value == 0u ||
      HNSW_GRAPH_HEADER + node_count * HNSW_NODE_GRAPH_ELEMENTS > HNSW_CHUNK_ELEMENTS) {
    return;
  }
  for (node = 0u; node < node_count; ++node) {
    unsigned node_level;
    if (!hnsw_exact_uint(graph[hnsw_graph_record(node)], HNSW_MAX_LEVELS - 1u, &node_level)) {
      return;
    }
    if (node == entry_point && node_level != max_level) {
      return;
    }
    for (layer = 0u; layer < HNSW_MAX_LEVELS; ++layer) {
      unsigned list = hnsw_graph_layer(node, layer);
      unsigned neighbor_count;
      if (!hnsw_exact_uint(graph[list], HNSW_MAX_M, &neighbor_count) ||
          (layer > node_level && neighbor_count != 0u)) {
        return;
      }
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor;
        unsigned neighbor_level;
        if (!hnsw_exact_uint(graph[list + 1u + slot], node_count - 1u, &neighbor) ||
            neighbor == node ||
            !hnsw_exact_uint(graph[hnsw_graph_record(neighbor)], HNSW_MAX_LEVELS - 1u,
                             &neighbor_level) ||
            neighbor_level < layer) {
          return;
        }
      }
    }
  }

  current = entry_point;
  current_distance = hnsw_distance(query, vectors, dimensions, current);
  distance_count = 1u;
  for (layer = max_level; layer > 0u; --layer) {
    unsigned step;
    for (step = 0u; step < node_count; ++step) {
      unsigned list = hnsw_graph_layer(current, layer);
      unsigned neighbor_count = (unsigned)graph[list];
      unsigned best = current;
      float best_distance = current_distance;
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor = (unsigned)graph[list + 1u + slot];
        float distance = hnsw_distance(query, vectors, dimensions, neighbor);
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
      hnsw_write_status(output, HNSW_STATUS_INTERNAL_BOUND);
      return;
    }
  }

  for (node = 0u; node < HNSW_MAX_NODES; ++node) {
    visited[node] = 0u;
  }
  visited[current] = 1u;
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
      unsigned neighbor_count = (unsigned)graph[list];
      for (slot = 0u; slot < neighbor_count; ++slot) {
        unsigned neighbor = (unsigned)graph[list + 1u + slot];
        if (visited[neighbor] == 0u) {
          float distance;
          visited[neighbor] = 1u;
          ++visited_count;
          distance = hnsw_distance(query, vectors, dimensions, neighbor);
          ++distance_count;
          if (top_size < ef_search ||
              hnsw_better(distance, neighbor, top_distances[0], top_nodes[0])) {
            if (candidate_size >= HNSW_MAX_NODES) {
              hnsw_write_status(output, HNSW_STATUS_INTERNAL_BOUND);
              return;
            }
            hnsw_candidate_push(candidate_distances, candidate_nodes, &candidate_size, distance,
                                neighbor);
            hnsw_top_push(top_distances, top_nodes, &top_size, distance, neighbor);
            if (top_size > ef_search) {
              hnsw_top_pop(top_distances, top_nodes, &top_size);
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
    hnsw_write_status(output, HNSW_STATUS_BAD_SEARCH);
    return;
  }
  output[0] = (float)HNSW_OUTPUT_MAGIC;
  output[1] = (float)HNSW_STATUS_OK;
  output[2] = (float)node_count;
  output[3] = (float)dimensions;
  output[4] = (float)entry_point;
  output[5] = (float)max_level;
  output[6] = (float)ef_search;
  output[7] = (float)result_count;
  output[8] = (float)visited_count;
  output[9] = (float)distance_count;
  output[10] = (float)upper_hops;
  output[11] = (float)base_expansions;
  output[12] = (float)terminated_by_bound;
  output[13] = (float)candidate_peak;
  output[14] = (float)top_peak;
  output[15] = (float)result_count;

  while (top_size > result_count) {
    hnsw_top_pop(top_distances, top_nodes, &top_size);
  }
  for (value = result_count; value > 0u; --value) {
    unsigned output_index = HNSW_OUTPUT_HEADER + (value - 1u) * 2u;
    output[output_index] = (float)top_nodes[0];
    output[output_index + 1u] = top_distances[0];
    hnsw_top_pop(top_distances, top_nodes, &top_size);
  }
}
