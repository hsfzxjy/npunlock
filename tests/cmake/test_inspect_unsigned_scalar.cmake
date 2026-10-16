foreach(dtype IN ITEMS u16 u32)
  execute_process(
    COMMAND "${INSPECT}" --graph "${FIXTURE_ROOT}/subtract-${dtype}-scalar-1x32.blob"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${dtype} scalar inspection failed: ${error}")
  endif()
  if(NOT output MATCHES "\"precision\": \"${dtype}\"")
    message(FATAL_ERROR "${dtype} scalar inspection omitted its precision")
  endif()
  if(NOT output MATCHES "\"input_1_scalar\": true")
    message(FATAL_ERROR "${dtype} scalar inspection omitted its scalar contract")
  endif()
endforeach()
