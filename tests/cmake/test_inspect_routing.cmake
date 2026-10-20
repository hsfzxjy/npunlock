execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}"
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first_output
  ERROR_VARIABLE first_error
)
if(NOT first_result EQUAL 0)
  message(FATAL_ERROR "first routing inspection failed: ${first_error}")
endif()

execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}"
  RESULT_VARIABLE second_result
  OUTPUT_VARIABLE second_output
  ERROR_VARIABLE second_error
)
if(NOT second_result EQUAL 0)
  message(FATAL_ERROR "second routing inspection failed: ${second_error}")
endif()
if(NOT first_output STREQUAL second_output)
  message(FATAL_ERROR "routing inspection is not deterministic")
endif()

foreach(required
    "\"schema\": \"npunlock.porting.inspect.v17\""
    "\"code_observation\""
    "\"offset_raw\""
    "\"extent\""
    "\"sha256\""
    "\"tile_count_observed\""
    "\"tile_candidate_raw\""
    "\"parameter_base\""
    "\"wait_mask_raw\""
    "\"post_mask_raw\""
    "\"barrier_group_raw\""
    "\"barrier_mask_raw\""
    "\"start_after_raw\""
    "\"clean_after_raw\""
    "\"act_edge_candidates\""
    "\"act_consumer_cover_observations\""
    "\"data_symbol_raw\""
    "\"data_addend_raw\""
    "\"dimensions\""
    "\"strides_bits\""
    "\"network_output_routes\""
    "\"output_slice_candidates\""
    "\"route_candidate_count\""
    "\"output_slice_addend_candidate\""
    "\"output_cover_observations\""
    "\"act_group_count\""
    "\"routed_invocation_count\""
    "\"unrouted_invocation_count\""
    "\"exact_cover\""
    "\"replica_group_observations\""
    "\"replica_candidates\""
    "\"group_completeness_observations\""
    "\"replica_invocation_count\""
    "\"complete\""
    "\"cmx_source_addend_raw\""
    "\"output_addend_raw\""
    "\"length_bytes\""
    "\"dpu\""
    "\"invariant_count\": 1"
    "\"input_cmx_relocations\""
    "\"output_cmx_relocations\""
    "\"register_offset\""
    "\"variant_count_raw\""
    "\"input_tensor_observation\""
    "\"output_tensor_observation\""
    "\"dimensions_xyz\""
    "\"act_dpu_edge_candidates\""
    "\"dpu_section_index\""
    "\"dpu_field_mask_raw\""
    "\"act_candidate_count\""
    "\"act_dpu_output_cover_observations\""
    "candidates do not classify unrouted invocations as replicas")
  if(NOT first_output MATCHES "${required}")
    message(FATAL_ERROR "routing inspection omitted required observation: ${required}")
  endif()
endforeach()

foreach(expected
    "\"code_observation\": {\"offset_raw\": 0, \"extent\": 5280, \"sha256\": \"bcb89071968d07af5e42e824beb91467a60e50dd64c6414a71b2d04875dd886f\"}"
    "\"register_offset\": 92, \"relocation_type\": 4, \"data_symbol_raw\": 0, \"data_addend_raw\": 768"
    "\"register_offset\": 160, \"relocation_type\": 7, \"data_symbol_raw\": 0, \"data_addend_raw\": 832"
    "\"cluster_raw\": 0, \"wait_mask_raw\": 1, \"post_mask_raw\": 2"
    "\"variant_count_raw\": 1"
    "\"input_tensor_observation\": {\"dimensions_xyz\": [1, 1, 16], \"dtype_raw\": 0, \"dense_raw\": 1, \"element_count\": 16, \"span_bytes\": 32}"
    "\"output_tensor_observation\": {\"dimensions_xyz\": [1, 1, 16], \"dtype_raw\": 0, \"sparse_raw\": 0, \"element_count\": 16, \"span_bytes\": 32}")
  string(FIND "${first_output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "DPU invariant routing changed a retained observation: ${expected}")
  endif()
endforeach()

foreach(expected
    "\"direction\": \"dpu_to_act\", \"act_group_index\": 0, \"act_invocation_index\": 0, \"act_tensor_record_index\": 0, \"dpu_invariant_index\": 0, \"dpu_field_mask_raw\": 3, \"act_candidate_count\": 1"
    "\"act_data_addend_raw\": 832, \"act_span_bytes\": 16, \"dpu_data_addend_raw\": 832, \"dpu_span_bytes\": 32, \"barrier_intersection_raw\": 2"
    "\"direction\": \"dpu_to_act\", \"act_group_index\": 0, \"act_invocation_index\": 1, \"act_tensor_record_index\": 0, \"dpu_invariant_index\": 0, \"dpu_field_mask_raw\": 3, \"act_candidate_count\": 1"
    "\"act_data_addend_raw\": 848, \"act_span_bytes\": 16, \"dpu_data_addend_raw\": 832, \"dpu_span_bytes\": 32, \"barrier_intersection_raw\": 2")
  string(FIND "${first_output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT/DPU candidate changed a retained observation: ${expected}")
  endif()
endforeach()

foreach(expected
    "\"parameter_base\": 0"
    "\"parameter_base\": 192"
    "\"wait_mask_raw\": 2"
    "\"post_mask_raw\": 4"
    "\"barrier_group_raw\": 1"
    "\"barrier_mask_raw\": 2"
    "\"start_after_raw\": 3"
    "\"clean_after_raw\": 2"
    "\"data_addend_raw\": 832"
    "\"data_addend_raw\": 848"
    "\"data_addend_raw\": 0"
    "\"data_addend_raw\": 16"
    "\"dimensions\": [8, 1, 1, 1]"
    "\"strides_bits\": [16, 256, 256, 256]")
  string(FIND "${first_output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "routing inspection changed a retained observation: ${expected}")
  endif()
endforeach()

foreach(expected
    "\"cmx_source_addend_raw\": 0"
    "\"output_addend_raw\": 0"
    "\"length_bytes\": 32")
  string(FIND "${first_output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "network-output routing changed a retained observation: ${expected}")
  endif()
endforeach()
