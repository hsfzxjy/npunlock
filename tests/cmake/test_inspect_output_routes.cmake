execute_process(
  COMMAND "${INSPECTOR}" --graph "${GRAPH}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "output-route inspection failed: ${error}")
endif()

string(REGEX MATCHALL "\"cmx_source_addend_raw\"" route_entries "${output}")
list(LENGTH route_entries route_count)
if(NOT route_count EQUAL 2)
  message(FATAL_ERROR "expected two network-output routes, got ${route_count}")
endif()

string(REGEX MATCHALL "\"output_slice_addend_candidate\"" slice_entries "${output}")
list(LENGTH slice_entries slice_count)
if(NOT slice_count EQUAL 4)
  message(FATAL_ERROR "expected four ACT output-slice candidates, got ${slice_count}")
endif()

foreach(expected
    "{\"dma_section_index\": 5, \"dma_task_index\": 2, \"cmx_source_symbol_raw\": 0, \"cmx_source_addend_raw\": 64, \"output_symbol_raw\": 1, \"output_addend_raw\": 0, \"length_bytes\": 32}"
    "{\"dma_section_index\": 6, \"dma_task_index\": 1, \"cmx_source_symbol_raw\": 0, \"cmx_source_addend_raw\": 2097216, \"output_symbol_raw\": 1, \"output_addend_raw\": 32, \"length_bytes\": 32}")
  string(FIND "${output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "network-output route changed a retained observation: ${expected}")
  endif()
endforeach()

foreach(expected
    "\"act_invocation_index\": 0, \"act_tensor_record_index\": 2, \"dma_section_index\": 5, \"dma_task_index\": 2, \"route_candidate_count\": 1"
    "\"act_data_addend_raw\": 64, \"act_span_bytes\": 16, \"route_cmx_source_addend_raw\": 64, \"route_length_bytes\": 32, \"output_symbol_raw\": 1, \"route_output_addend_raw\": 0, \"output_slice_addend_candidate\": 0"
    "\"act_invocation_index\": 1, \"act_tensor_record_index\": 2, \"dma_section_index\": 6, \"dma_task_index\": 1, \"route_candidate_count\": 1"
    "\"act_data_addend_raw\": 2097216, \"act_span_bytes\": 16, \"route_cmx_source_addend_raw\": 2097216, \"route_length_bytes\": 32, \"output_symbol_raw\": 1, \"route_output_addend_raw\": 32, \"output_slice_addend_candidate\": 32"
    "\"act_invocation_index\": 2, \"act_tensor_record_index\": 2, \"dma_section_index\": 5, \"dma_task_index\": 2, \"route_candidate_count\": 1"
    "\"act_data_addend_raw\": 80, \"act_span_bytes\": 16, \"route_cmx_source_addend_raw\": 64, \"route_length_bytes\": 32, \"output_symbol_raw\": 1, \"route_output_addend_raw\": 0, \"output_slice_addend_candidate\": 16"
    "\"act_invocation_index\": 3, \"act_tensor_record_index\": 2, \"dma_section_index\": 6, \"dma_task_index\": 1, \"route_candidate_count\": 1"
    "\"act_data_addend_raw\": 2097232, \"act_span_bytes\": 16, \"route_cmx_source_addend_raw\": 2097216, \"route_length_bytes\": 32, \"output_symbol_raw\": 1, \"route_output_addend_raw\": 32, \"output_slice_addend_candidate\": 48")
  string(FIND "${output}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT output-slice candidate changed: ${expected}")
  endif()
endforeach()
