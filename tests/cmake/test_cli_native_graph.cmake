if(NOT DEFINED NPURUN OR NOT DEFINED FIXTURE_ROOT OR NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "NPURUN, FIXTURE_ROOT, and OUTPUT_DIR are required")
endif()

set(output "${OUTPUT_DIR}/cli-native-graph.blob")
set(manifest "${OUTPUT_DIR}/cli-native-graph.json")
file(REMOVE "${output}" "${manifest}")

execute_process(
  COMMAND
    "${NPURUN}" build
    --native-graph "${FIXTURE_ROOT}/abs-add-1x16-tile1.blob"
    --shave-elf "${FIXTURE_ROOT}/add1-fp16.elf"
    --patch-position 0
    --output "${output}"
    --manifest "${manifest}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(NOT result EQUAL 0)
  message(FATAL_ERROR "offline npurun build failed:\n${stderr}\n${stdout}")
endif()

file(SHA256 "${output}" output_hash)
file(SHA256 "${FIXTURE_ROOT}/add1-shared-1x16.blob" expected_hash)
if(NOT output_hash STREQUAL expected_hash)
  message(FATAL_ERROR "offline npurun output does not match the retained fixture")
endif()

file(READ "${manifest}" manifest_json)
string(FIND "${manifest_json}" "\"carrier_graph\": {\"source\": \"caller\"" carrier)
string(FIND "${manifest_json}" "\"ir2blob\": null" skipped)
string(FIND "${manifest_json}" "${output_hash}" patched)
if(carrier EQUAL -1 OR skipped EQUAL -1 OR patched EQUAL -1)
  message(FATAL_ERROR "offline npurun manifest lacks expected provenance:\n${manifest_json}")
endif()

file(REMOVE "${output}" "${manifest}")
