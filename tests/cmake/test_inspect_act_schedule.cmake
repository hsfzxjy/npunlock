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
    "\"schema\": \"npunlock.porting.inspect.v6\""
    "\"invocation_index\": 0"
    "\"wait_mask_raw\": 1, \"post_mask_raw\": 2"
    "\"data_addend_raw\": 64"
    "\"invocation_index\": 4"
    "\"wait_mask_raw\": 2, \"post_mask_raw\": 4"
    "\"producer_invocation_index\": 0, \"producer_tensor_record_index\": 1, \"consumer_group_index\": 1, \"consumer_invocation_index\": 4"
    "\"producer_invocation_index\": 3, \"producer_tensor_record_index\": 1, \"consumer_group_index\": 1, \"consumer_invocation_index\": 7"
    "\"consumer_candidate_count\": 1"
    "\"barrier_intersection_raw\": 2")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT schedule inspection changed a retained observation: ${expected}")
  endif()
endforeach()

string(REGEX MATCHALL "\"producer_group_index\"" candidate_entries "${report}")
list(LENGTH candidate_entries candidate_count)
if(NOT candidate_count EQUAL 4)
  message(FATAL_ERROR "expected four ACT edge candidates, found ${candidate_count}")
endif()
