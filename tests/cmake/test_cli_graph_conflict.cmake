if(NOT DEFINED NPURUN)
  message(FATAL_ERROR "NPURUN is required")
endif()

execute_process(
  COMMAND "${NPURUN}" build --ir model.xml --native-graph carrier.blob
          --shave-elf kernel.elf
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(result EQUAL 0)
  message(FATAL_ERROR "npurun accepted conflicting graph inputs")
endif()

string(FIND "${stderr}" "exactly one of --ir or --native-graph" expected)
if(expected EQUAL -1)
  message(FATAL_ERROR "npurun returned an unexpected error:\n${stderr}\n${stdout}")
endif()

execute_process(
  COMMAND "${NPURUN}" build --native-graph carrier.blob --build-flags flags
          --shave-elf kernel.elf
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(result EQUAL 0)
  message(FATAL_ERROR "npurun accepted IR compiler options with a native graph")
endif()

string(FIND "${stderr}" "--native-graph cannot be combined with IR compiler options" expected)
if(expected EQUAL -1)
  message(FATAL_ERROR "npurun returned an unexpected error:\n${stderr}\n${stdout}")
endif()
