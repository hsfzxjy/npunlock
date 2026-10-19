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
    "\"schema\": \"npunlock.porting.inspect.v5\""
    "\"invocation_index\": 0"
    "\"wait_mask_raw\": 1, \"post_mask_raw\": 2"
    "\"data_addend_raw\": 64"
    "\"invocation_index\": 4"
    "\"wait_mask_raw\": 2, \"post_mask_raw\": 4")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT schedule inspection changed a retained observation: ${expected}")
  endif()
endforeach()
