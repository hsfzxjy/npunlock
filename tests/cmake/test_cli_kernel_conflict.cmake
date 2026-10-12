if(NOT DEFINED NPURUN)
  message(FATAL_ERROR "NPURUN is required")
endif()

execute_process(
  COMMAND "${NPURUN}" build --shave-source kernel.c --shave-elf kernel.elf
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(result EQUAL 0)
  message(FATAL_ERROR "npurun accepted conflicting kernel inputs")
endif()

string(FIND "${stderr}" "exactly one of --shave-source or --shave-elf" expected)
if(expected EQUAL -1)
  message(FATAL_ERROR "npurun returned an unexpected error:\n${stderr}\n${stdout}")
endif()
