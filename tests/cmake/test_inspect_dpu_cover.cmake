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
  message(FATAL_ERROR "ACT/DPU output-cover inspection failed: ${error}")
endif()

foreach(expected
    "\"schema\": \"npunlock.porting.inspect.v23\""
    "\"precision\": \"mixed\""
    "\"dpu_section_index\": 14"
    "\"act_dpu_output_cover_observations\""
    "\"act_group_index\": 0, \"invocation_count\": 2, \"matched_invocation_count\": 2, \"unresolved_invocation_count\": 0, \"dpu_input_count\": 2, \"exact_cover_input_count\": 2, \"exact_cover\": true, \"unresolved_flags\": 0")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT/DPU output-cover report omitted: ${expected}")
  endif()
endforeach()
