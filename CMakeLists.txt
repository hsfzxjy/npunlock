cmake_minimum_required(VERSION 3.24)

project(npunlock VERSION 0.1.0 LANGUAGES C)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NPUNLOCK_BUILD_TESTS "Build offline unit tests" ON)
option(NPUNLOCK_ENABLE_NPU_TESTS "Enable tests that require an Intel NPU" OFF)
option(NPUNLOCK_ENABLE_MOVITOOLS_TESTS "Enable tests that invoke caller-supplied MoviTools" OFF)
option(NPUNLOCK_ENABLE_PATCHBLOB_TESTS "Enable tests using caller-supplied retained graph fixtures" OFF)
set(NPUNLOCK_MOVITOOLS_DIR "" CACHE PATH "Directory containing the three MoviTools DLLs")
set(NPUNLOCK_MOVITOOLS_TEST_SOURCE "" CACHE FILEPATH "C source for the MoviTools integration test")
set(NPUNLOCK_MOVITOOLS_TEST_LINKER_SCRIPT "" CACHE FILEPATH
    "Linker script for the MoviTools integration test")
set(NPUNLOCK_MOVITOOLS_TEST_EXPECTED_ELF "" CACHE FILEPATH
    "Expected repeatable ELF for the MoviTools integration test")
set(NPUNLOCK_NPU_TEST_IR_XML "" CACHE FILEPATH "IR XML for the NPU integration test")
set(NPUNLOCK_NPU_TEST_WEIGHTS "" CACHE FILEPATH "Weights for the NPU integration test")
set(NPUNLOCK_NPU_TEST_EXPECTED_BLOB "" CACHE FILEPATH
    "Expected repeatable native blob for the NPU integration test")
set(NPUNLOCK_PATCHBLOB_TEST_CARRIER "" CACHE FILEPATH "ACT carrier graph for patchblob testing")
set(NPUNLOCK_PATCHBLOB_TEST_ELF "" CACHE FILEPATH "Linked SHAVE ELF for patchblob testing")
set(NPUNLOCK_PATCHBLOB_TEST_EXPECTED_BLOB "" CACHE FILEPATH
    "Expected patched graph for patchblob testing")

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(npunlock_warnings INTERFACE)
if(MSVC)
  target_compile_options(npunlock_warnings INTERFACE /W4 /WX /permissive-)
else()
  target_compile_options(npunlock_warnings INTERFACE -Wall -Wextra -Werror -Wpedantic)
endif()

add_library(npunlock_common
  src/common/buffer.c
  src/common/checked.c
  src/common/diagnostic.c
  src/common/elf32.c
  src/common/sha256.c
  src/common/status.c
)
add_library(npunlock::common ALIAS npunlock_common)
target_include_directories(npunlock_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE src/common
)
target_link_libraries(npunlock_common PRIVATE $<BUILD_INTERFACE:npunlock_warnings>)

if(WIN32)
  set(NPUNLOCK_MOVI_STAGE_SOURCE src/shavecc/movi_stage_win32.c)
  set(NPUNLOCK_IR_WORKER_SOURCE src/ir2blob/ir_worker_win32.c)
  set(NPUNLOCK_RUN_WORKER_SOURCE src/npurun/run_worker_win32.c)
  add_executable(npunlock_movi_worker src/workers/movi_worker.c)
  target_include_directories(npunlock_movi_worker PRIVATE src/workers)
  target_link_libraries(npunlock_movi_worker PRIVATE npunlock_warnings)
  add_executable(npunlock_ir_worker src/workers/ir_worker.c)
  target_include_directories(npunlock_ir_worker PRIVATE src/workers)
  target_link_libraries(npunlock_ir_worker PRIVATE npunlock_warnings)
  add_executable(npunlock_run_worker src/workers/run_worker.c)
  target_include_directories(npunlock_run_worker PRIVATE src/workers)
  target_link_libraries(npunlock_run_worker PRIVATE npunlock_warnings)
else()
  set(NPUNLOCK_MOVI_STAGE_SOURCE src/shavecc/movi_stage_stub.c)
  set(NPUNLOCK_IR_WORKER_SOURCE src/ir2blob/ir_worker_stub.c)
  set(NPUNLOCK_RUN_WORKER_SOURCE src/npurun/run_worker_stub.c)
endif()

add_library(shavecc src/shavecc/shavecc.c ${NPUNLOCK_MOVI_STAGE_SOURCE})
add_library(npunlock::shavecc ALIAS shavecc)
target_include_directories(shavecc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(shavecc PRIVATE src/common)
target_include_directories(shavecc PRIVATE src src/shavecc)
target_link_libraries(shavecc PRIVATE npunlock_common $<BUILD_INTERFACE:npunlock_warnings>)

add_library(ir2blob src/ir2blob/ir2blob.c ${NPUNLOCK_IR_WORKER_SOURCE})
add_library(npunlock::ir2blob ALIAS ir2blob)
target_include_directories(ir2blob PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(ir2blob PRIVATE src/common)
target_include_directories(ir2blob PRIVATE src src/ir2blob)
target_link_libraries(ir2blob PRIVATE npunlock_common $<BUILD_INTERFACE:npunlock_warnings>)

add_library(patchblob src/patchblob/graph_blob.c src/patchblob/patchblob.c)
add_library(npunlock::patchblob ALIAS patchblob)
target_include_directories(patchblob PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(patchblob PRIVATE src/common src/patchblob)
target_link_libraries(patchblob PRIVATE npunlock_common $<BUILD_INTERFACE:npunlock_warnings>)

target_compile_definitions(npunlock_common PRIVATE NPUNLOCK_COMMON_BUILDING)
target_compile_definitions(shavecc PRIVATE NPUNLOCK_SHAVECC_BUILDING)
target_compile_definitions(ir2blob PRIVATE NPUNLOCK_IR2BLOB_BUILDING)
target_compile_definitions(patchblob PRIVATE NPUNLOCK_PATCHBLOB_BUILDING)
foreach(target npunlock_common shavecc ir2blob patchblob)
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(${target} PUBLIC NPUNLOCK_SHARED)
  endif()
endforeach()

add_executable(npurun src/npurun/main.c ${NPUNLOCK_RUN_WORKER_SOURCE})
target_include_directories(npurun PRIVATE src src/common src/npurun)
target_link_libraries(npurun
  PRIVATE shavecc ir2blob patchblob npunlock_common $<BUILD_INTERFACE:npunlock_warnings>
)

set(NPUNLOCK_FORMAT_SOURCES
  include/npunlock/api.h
  include/npunlock/buffer.h
  include/npunlock/error.h
  include/npunlock/ir2blob.h
  include/npunlock/patchblob.h
  include/npunlock/shavecc.h
  include/npunlock/version.h
  src/common/buffer.c
  src/common/checked.c
  src/common/diagnostic.c
  src/common/elf32.c
  src/common/elf32.h
  src/common/internal.h
  src/common/sha256.c
  src/common/status.c
  src/ir2blob/ir2blob.c
  src/ir2blob/ir_worker.h
  src/ir2blob/ir_worker_stub.c
  src/ir2blob/ir_worker_win32.c
  src/npurun/main.c
  src/npurun/run_worker.h
  src/npurun/run_worker_stub.c
  src/npurun/run_worker_win32.c
  src/patchblob/patchblob.c
  src/patchblob/graph_blob.c
  src/patchblob/graph_blob.h
  src/shavecc/shavecc.c
  src/shavecc/movi_stage.h
  src/shavecc/movi_stage_stub.c
  src/shavecc/movi_stage_win32.c
  src/workers/movi_protocol.h
  src/workers/movi_worker.c
  src/workers/ir_protocol.h
  src/workers/ir_worker.c
  src/workers/level_zero_min.h
  src/workers/run_protocol.h
  src/workers/run_worker.c
  tests/package/smoke.c
  tests/integration/test_shavecc.c
  tests/integration/test_ir2blob.c
  tests/integration/test_patchblob.c
  tests/unit/test_api.c
  tests/unit/test_common.c
  tests/unit/test_elf32.c
  tests/unit/test_graph_blob.c
)

find_program(CLANG_FORMAT_EXECUTABLE
  NAMES clang-format clang-format-19 clang-format-18
  HINTS
    "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/x64/bin"
    "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/bin"
)
if(CLANG_FORMAT_EXECUTABLE)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${NPUNLOCK_FORMAT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
  )
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror ${NPUNLOCK_FORMAT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
  )
else()
  add_custom_target(format
    COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found; install it to format C sources"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM
  )
  add_custom_target(format-check
    COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found; install it to check C formatting"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM
  )
endif()

if(BUILD_TESTING AND NPUNLOCK_BUILD_TESTS)
  add_executable(npunlock_test_common tests/unit/test_common.c)
  target_include_directories(npunlock_test_common PRIVATE src/common)
  target_link_libraries(npunlock_test_common PRIVATE npunlock_common npunlock_warnings)
  add_test(NAME common COMMAND npunlock_test_common)

  add_executable(npunlock_test_api tests/unit/test_api.c)
  if(WIN32)
    add_dependencies(npunlock_test_api npunlock_movi_worker npunlock_ir_worker npunlock_run_worker)
  endif()
  target_link_libraries(
    npunlock_test_api PRIVATE shavecc ir2blob patchblob npunlock_common npunlock_warnings
  )
  add_test(NAME api_contract COMMAND npunlock_test_api)

  add_executable(npunlock_test_elf32 tests/unit/test_elf32.c)
  target_include_directories(npunlock_test_elf32 PRIVATE src/common)
  target_link_libraries(npunlock_test_elf32 PRIVATE npunlock_common npunlock_warnings)
  add_test(NAME shave_elf_parser COMMAND npunlock_test_elf32)

  add_executable(
    npunlock_test_graph_blob tests/unit/test_graph_blob.c src/patchblob/graph_blob.c
  )
  target_include_directories(npunlock_test_graph_blob PRIVATE src/common src/patchblob)
  target_link_libraries(
    npunlock_test_graph_blob PRIVATE npunlock_common npunlock_warnings
  )
  add_test(NAME graph_blob_rejection COMMAND npunlock_test_graph_blob)

  add_test(NAME cli_version COMMAND npurun --version)
  set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "npunlock 0.1.0")
  add_test(NAME cli_help COMMAND npurun --help)
  set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "npurun.*build")

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy"
      RESULT_VARIABLE NPUNLOCK_NUMPY_IMPORT_RESULT
    )
    if(NPUNLOCK_NUMPY_IMPORT_RESULT EQUAL 0)
      add_test(
        NAME python_api
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_python_api.py
      )
    endif()
  endif()

  if(NPUNLOCK_ENABLE_PATCHBLOB_TESTS)
    foreach(required_file
        NPUNLOCK_PATCHBLOB_TEST_CARRIER
        NPUNLOCK_PATCHBLOB_TEST_ELF
        NPUNLOCK_PATCHBLOB_TEST_EXPECTED_BLOB)
      if(NOT IS_ABSOLUTE "${${required_file}}" OR NOT EXISTS "${${required_file}}")
        message(FATAL_ERROR "${required_file} must name an existing absolute file")
      endif()
    endforeach()
    add_executable(npunlock_test_patchblob_integration tests/integration/test_patchblob.c)
    target_link_libraries(
      npunlock_test_patchblob_integration PRIVATE patchblob npunlock_common npunlock_warnings
    )
    add_test(
      NAME patchblob_shared_add1
      COMMAND npunlock_test_patchblob_integration
        "${NPUNLOCK_PATCHBLOB_TEST_CARRIER}"
        "${NPUNLOCK_PATCHBLOB_TEST_ELF}"
        "${NPUNLOCK_PATCHBLOB_TEST_EXPECTED_BLOB}"
    )
  endif()

  if(NPUNLOCK_ENABLE_MOVITOOLS_TESTS)
    if(NOT WIN32)
      message(FATAL_ERROR "MoviTools integration tests require Windows")
    endif()
    foreach(required_path
        NPUNLOCK_MOVITOOLS_DIR
        NPUNLOCK_MOVITOOLS_TEST_SOURCE
        NPUNLOCK_MOVITOOLS_TEST_LINKER_SCRIPT
        NPUNLOCK_MOVITOOLS_TEST_EXPECTED_ELF)
      if(NOT IS_ABSOLUTE "${${required_path}}" OR NOT EXISTS "${${required_path}}")
        message(FATAL_ERROR "${required_path} must name an existing absolute path")
      endif()
    endforeach()
    foreach(required_dll moviCompile64.dll moviAsm64.dll moviLLD64.dll)
      if(NOT EXISTS "${NPUNLOCK_MOVITOOLS_DIR}/${required_dll}")
        message(FATAL_ERROR "missing ${required_dll} under NPUNLOCK_MOVITOOLS_DIR")
      endif()
    endforeach()
    add_executable(npunlock_test_shavecc_integration tests/integration/test_shavecc.c)
    target_link_libraries(
      npunlock_test_shavecc_integration PRIVATE shavecc npunlock_common npunlock_warnings
    )
    add_dependencies(npunlock_test_shavecc_integration npunlock_movi_worker)
    add_test(
      NAME movitools_add1
      COMMAND npunlock_test_shavecc_integration
        "${NPUNLOCK_MOVITOOLS_DIR}"
        "${NPUNLOCK_MOVITOOLS_TEST_SOURCE}"
        "${NPUNLOCK_MOVITOOLS_TEST_LINKER_SCRIPT}"
        "${NPUNLOCK_MOVITOOLS_TEST_EXPECTED_ELF}"
    )
    set_tests_properties(movitools_add1 PROPERTIES TIMEOUT 30)
  endif()

  if(NPUNLOCK_ENABLE_NPU_TESTS)
    if(NOT WIN32)
      message(FATAL_ERROR "NPU integration tests require Windows")
    endif()
    foreach(required_file
        NPUNLOCK_NPU_TEST_IR_XML
        NPUNLOCK_NPU_TEST_WEIGHTS
        NPUNLOCK_NPU_TEST_EXPECTED_BLOB)
      if(NOT IS_ABSOLUTE "${${required_file}}" OR NOT EXISTS "${${required_file}}")
        message(FATAL_ERROR "${required_file} must name an existing absolute file")
      endif()
    endforeach()
    add_executable(npunlock_test_ir2blob_integration tests/integration/test_ir2blob.c)
    target_link_libraries(
      npunlock_test_ir2blob_integration PRIVATE ir2blob npunlock_common npunlock_warnings
    )
    add_dependencies(npunlock_test_ir2blob_integration npunlock_ir_worker)
    add_test(
      NAME npu_ir2blob_abs
      COMMAND npunlock_test_ir2blob_integration
        "${NPUNLOCK_NPU_TEST_IR_XML}"
        "${NPUNLOCK_NPU_TEST_WEIGHTS}"
        "${NPUNLOCK_NPU_TEST_EXPECTED_BLOB}"
    )
    set_tests_properties(npu_ir2blob_abs PROPERTIES TIMEOUT 30)
  endif()
endif()

set(NPUNLOCK_INSTALL_TARGETS npunlock_common shavecc ir2blob patchblob npurun)
if(WIN32)
  list(APPEND NPUNLOCK_INSTALL_TARGETS npunlock_movi_worker npunlock_ir_worker npunlock_run_worker)
endif()
install(TARGETS ${NPUNLOCK_INSTALL_TARGETS}
  EXPORT npunlockTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT npunlockTargets
  NAMESPACE npunlock::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/npunlock
)

configure_package_config_file(
  cmake/npunlockConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/npunlockConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/npunlock
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/npunlockConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/npunlockConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/npunlockConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/npunlock
)
