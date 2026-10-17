foreach(dtype IN ITEMS fp16 u32)
  execute_process(
    COMMAND "${INSPECT}" --graph
      "${FIXTURE_ROOT}/subtract-${dtype}-scalar-first-1x32.blob"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${dtype} scalar-first inspection failed: ${error}")
  endif()
  if(NOT output MATCHES "\"precision\": \"${dtype}\"")
    message(FATAL_ERROR "${dtype} scalar-first inspection omitted its precision")
  endif()
  if(NOT output MATCHES "\"scalar_input_indices\": [[]0[]]")
    message(FATAL_ERROR "${dtype} scalar-first inspection omitted input position zero")
  endif()
  if(NOT output MATCHES "\"input_1_scalar\": false")
    message(FATAL_ERROR "${dtype} scalar-first inspection misreported input position one")
  endif()
endforeach()
