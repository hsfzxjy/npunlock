if(NOT DEFINED INSPECTOR OR NOT DEFINED GRAPH OR NOT DEFINED DIFFERENT_GRAPH)
  message(FATAL_ERROR "INSPECTOR, GRAPH, and DIFFERENT_GRAPH are required")
endif()

execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}" --compare "${GRAPH}"
  RESULT_VARIABLE same_result
  OUTPUT_VARIABLE same_report
  ERROR_VARIABLE same_error
)
if(NOT same_result EQUAL 0)
  message(FATAL_ERROR "self comparison failed: ${same_error}")
endif()
foreach(expected
    "\"schema\": \"npunlock.porting.compare.v1\""
    "\"contract_stable\": true"
    "\"unique_changed_group\": false"
    "\"accepted\": false"
    "\"mismatch_flags\": 0"
    "\"changed_group_indices\": []")
  string(FIND "${same_report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "self comparison omitted: ${expected}")
  endif()
endforeach()

execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}" --compare "${DIFFERENT_GRAPH}"
  RESULT_VARIABLE different_result
  OUTPUT_VARIABLE different_report
  ERROR_VARIABLE different_error
)
if(NOT different_result EQUAL 0)
  message(FATAL_ERROR "different comparison failed: ${different_error}")
endif()
foreach(expected
    "\"contract_stable\": false"
    "\"accepted\": false"
    "\"structure\"")
  string(FIND "${different_report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "different comparison omitted: ${expected}")
  endif()
endforeach()
