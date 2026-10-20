if(NOT DEFINED INSPECTOR OR NOT DEFINED GRAPH)
  message(FATAL_ERROR "INSPECTOR and GRAPH are required")
endif()

execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE report
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "ACT schedule inspection failed: ${error}")
endif()

foreach(expected
    "\"schema\": \"npunlock.porting.inspect.v16\""
    "\"invocation_index\": 0"
    "\"wait_mask_raw\": 1, \"post_mask_raw\": 2"
    "\"data_addend_raw\": 64"
    "\"invocation_index\": 4"
    "\"wait_mask_raw\": 2, \"post_mask_raw\": 4"
    "\"producer_invocation_index\": 0, \"producer_tensor_record_index\": 1, \"consumer_group_index\": 1, \"consumer_invocation_index\": 4"
    "\"producer_invocation_index\": 3, \"producer_tensor_record_index\": 1, \"consumer_group_index\": 1, \"consumer_invocation_index\": 7"
    "\"consumer_candidate_count\": 1"
    "\"schedule_hop_count\": 1"
    "\"barrier_intersection_raw\": 2")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT schedule inspection changed a retained observation: ${expected}")
  endif()
endforeach()

string(FIND "${report}"
  "{\"act_group_index\": 1, \"output_symbol_raw\": 1, \"invocation_count\": 4, \"routed_invocation_count\": 2, \"replica_invocation_count\": 2, \"unresolved_invocation_count\": 0, \"complete\": true, \"unresolved_flags\": 0}"
  completeness_offset)
if(completeness_offset EQUAL -1)
  message(FATAL_ERROR "final ACT group was not classified complete")
endif()

foreach(expected
    "\"act_group_index\": 1, \"routed_invocation_count\": 2, \"unrouted_invocation_count\": 2, \"translation_solution_count\": 1, \"cmx_translation_candidate\": 2097152"
    "\"unrouted_invocation_index\": 5, \"routed_invocation_index\": 4, \"unrouted_tile_candidate_raw\": 1, \"routed_tile_candidate_raw\": 0, \"output_symbol_raw\": 1, \"output_slice_addend_candidate\": 0, \"output_slice_span_bytes\": 32"
    "\"unrouted_invocation_index\": 7, \"routed_invocation_index\": 6, \"unrouted_tile_candidate_raw\": 1, \"routed_tile_candidate_raw\": 0, \"output_symbol_raw\": 1, \"output_slice_addend_candidate\": 32, \"output_slice_span_bytes\": 32")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "replica correlation changed a retained observation: ${expected}")
  endif()
endforeach()

string(REGEX MATCHALL "\"producer_group_index\"" candidate_entries "${report}")
list(LENGTH candidate_entries candidate_count)
if(NOT candidate_count EQUAL 4)
  message(FATAL_ERROR "expected four ACT edge candidates, found ${candidate_count}")
endif()
