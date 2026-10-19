execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "output-route inspection failed: ${error}")
endif()

string(REGEX MATCHALL "\"dma_section_index\"" route_entries "${output}")
list(LENGTH route_entries route_count)
if(NOT route_count EQUAL 2)
  message(FATAL_ERROR "expected two network-output routes, got ${route_count}")
endif()

foreach(expected
    "{\"dma_section_index\": 5, \"dma_task_index\": 2, \"cmx_source_symbol_raw\": 0, \"cmx_source_addend_raw\": 64, \"output_symbol_raw\": 1, \"output_addend_raw\": 0, \"length_bytes\": 32}"
    "{\"dma_section_index\": 6, \"dma_task_index\": 1, \"cmx_source_symbol_raw\": 0, \"cmx_source_addend_raw\": 2097216, \"output_symbol_raw\": 1, \"output_addend_raw\": 32, \"length_bytes\": 32}")
  string(FIND "${output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "network-output route changed a retained observation: ${expected}")
  endif()
endforeach()
