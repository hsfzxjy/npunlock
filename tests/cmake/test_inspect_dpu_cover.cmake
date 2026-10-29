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
    "\"schema\": \"npunlock.porting.inspect.v26\""
    "\"kind\": \"input\", \"symbol_table_index\": 24, \"symbol_index\": 1, \"size_bytes\": 64, \"name\": \"input\""
    "\"kind\": \"output\", \"symbol_table_index\": 25, \"symbol_index\": 1, \"size_bytes\": 32, \"name\": \"output\""
    "\"precision\": \"mixed\""
    "\"dpu_section_index\": 14"
    "\"graph_input_act_cover_observations\""
    "\"act_group_index\": 0, \"act_tensor_record_index\": 0, \"input_symbol_table_index\": 24, \"input_symbol_raw\": 1, \"invocation_count\": 2, \"direct_dma_task_count\": 1, \"replica_invocation_count\": 1, \"source_span_bytes\": 64, \"exact_cover\": true, \"unresolved_flags\": 0"
    "\"kind\": \"graph_input_to_act_via_dma\", \"source\": {\"kind\": \"graph_input\", \"primary_index\": 1, \"secondary_index\": 24}"
    "\"act_dpu_output_cover_observations\""
    "\"act_group_index\": 0, \"invocation_count\": 2, \"matched_invocation_count\": 2, \"unresolved_invocation_count\": 0, \"dpu_input_count\": 2, \"exact_cover_input_count\": 2, \"exact_cover\": true, \"unresolved_flags\": 0")
  string(FIND "${report}" "${expected}" expected_offset)
  if(expected_offset EQUAL -1)
    message(FATAL_ERROR "ACT/DPU output-cover report omitted: ${expected}")
  endif()
endforeach()
