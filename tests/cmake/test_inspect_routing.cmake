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
    "\"schema\": \"npunlock.porting.inspect.v4\""
    "\"tile_count_observed\""
    "\"tile_candidate_raw\""
    "\"parameter_base\""
    "\"data_symbol_raw\""
    "\"data_addend_raw\""
    "\"dimensions\""
    "\"strides_bits\""
    "\"network_output_routes\""
    "\"cmx_source_addend_raw\""
    "\"output_addend_raw\""
    "\"length_bytes\""
    "not logical offsets, partitions, or replicas")
  if(NOT first_output MATCHES "${required}")
    message(FATAL_ERROR "routing inspection omitted required observation: ${required}")
  endif()
endforeach()

foreach(expected
    "\"parameter_base\": 0"
    "\"parameter_base\": 192"
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
