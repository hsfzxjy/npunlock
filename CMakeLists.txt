cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0091 NEW)
project(npunlock VERSION 0.1.0 LANGUAGES C)

if(NOT WIN32)
  message(FATAL_ERROR "npunlock currently supports Windows only")
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NPUNLOCK_BUILD_TESTS "Build offline unit tests" ON)
option(NPUNLOCK_ENABLE_NPU_TESTS "Enable tests that require an Intel NPU" OFF)
option(NPUNLOCK_ENABLE_GRAPHINFER_TESTS "Enable bundled graph inference tests on an Intel NPU" OFF)
option(NPUNLOCK_ENABLE_MOVITOOLS_TESTS "Enable tests that invoke caller-supplied MoviTools" OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(npunlock_warnings INTERFACE)
if(MSVC)
  target_compile_options(npunlock_warnings INTERFACE /W4 /WX /permissive-)
else()
  target_compile_options(npunlock_warnings INTERFACE -Wall -Wextra -Werror -Wpedantic)
endif()

add_library(npunlock_common_objects OBJECT
  src/common/buffer.c
  src/common/checked.c
  src/common/diagnostic.c
  src/common/elf32.c
  src/common/sha256.c
  src/common/status.c
  src/common/worker_process_win32.c
)
target_include_directories(npunlock_common_objects PRIVATE include src/common)
target_link_libraries(npunlock_common_objects PRIVATE npunlock_warnings)

add_executable(npunlock_worker
  src/workers/worker.c
  src/workers/worker_io.c
  src/workers/movi_worker.c
  src/workers/ir_worker.c
  src/workers/infer_worker.c
)
target_include_directories(npunlock_worker PRIVATE src/common src/workers)
target_link_libraries(npunlock_worker PRIVATE npunlock_warnings)

set(NPUNLOCK_DEFAULT_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/shavecc/shave_kernel.ld")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NPUNLOCK_DEFAULT_LINKER_SCRIPT}")
file(READ "${NPUNLOCK_DEFAULT_LINKER_SCRIPT}" NPUNLOCK_DEFAULT_LINKER_SCRIPT_HEX HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1,"
  NPUNLOCK_DEFAULT_LINKER_SCRIPT_BYTES "${NPUNLOCK_DEFAULT_LINKER_SCRIPT_HEX}"
)
configure_file(
  cmake/default_linker_script.c.in
  generated/default_linker_script.c
  @ONLY
)

set(NPUNLOCK_NPU3720_KERNEL_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/include/npunlock/npu3720_kernel.h"
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NPUNLOCK_NPU3720_KERNEL_HEADER}")
file(READ "${NPUNLOCK_NPU3720_KERNEL_HEADER}" NPUNLOCK_NPU3720_KERNEL_HEADER_HEX HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1,"
  NPUNLOCK_NPU3720_KERNEL_HEADER_BYTES "${NPUNLOCK_NPU3720_KERNEL_HEADER_HEX}"
)
configure_file(
  cmake/npu3720_kernel_header.c.in
  generated/npu3720_kernel_header.c
  @ONLY
)

add_library(npunlock_shavecc_objects OBJECT
  src/shavecc/shavecc.c
  src/shavecc/movi_stage_win32.c
  ${CMAKE_CURRENT_BINARY_DIR}/generated/default_linker_script.c
  ${CMAKE_CURRENT_BINARY_DIR}/generated/npu3720_kernel_header.c
)
target_include_directories(npunlock_shavecc_objects PRIVATE include src/common src src/shavecc)
target_link_libraries(npunlock_shavecc_objects PRIVATE npunlock_warnings)

add_library(npunlock_ir2blob_objects OBJECT
  src/ir2blob/ir2blob.c
  src/ir2blob/ir_worker_win32.c
)
target_include_directories(npunlock_ir2blob_objects PRIVATE include src/common src src/ir2blob)
target_link_libraries(npunlock_ir2blob_objects PRIVATE npunlock_warnings)

add_library(npunlock_patchblob_objects OBJECT
  src/patchblob/graph_blob.c
  src/patchblob/patchblob.c
)
target_include_directories(npunlock_patchblob_objects PRIVATE include src/common src/patchblob)
target_link_libraries(npunlock_patchblob_objects PRIVATE npunlock_warnings)

add_library(npunlock_graphinfer_objects OBJECT
  src/graphinfer/graphinfer.c
  src/graphinfer/infer_worker_win32.c
  src/graphinfer/session_win32.c
)
target_include_directories(npunlock_graphinfer_objects PRIVATE include src src/common src/graphinfer)
target_link_libraries(npunlock_graphinfer_objects PRIVATE npunlock_warnings)

set(NPUNLOCK_OBJECT_TARGETS
  npunlock_common_objects
  npunlock_shavecc_objects
  npunlock_ir2blob_objects
  npunlock_patchblob_objects
  npunlock_graphinfer_objects
)
foreach(target IN LISTS NPUNLOCK_OBJECT_TARGETS)
  target_compile_definitions(${target} PRIVATE NPUNLOCK_BUILDING NPUNLOCK_SHARED)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(npunlock_native SHARED
  $<TARGET_OBJECTS:npunlock_common_objects>
  $<TARGET_OBJECTS:npunlock_shavecc_objects>
  $<TARGET_OBJECTS:npunlock_ir2blob_objects>
  $<TARGET_OBJECTS:npunlock_patchblob_objects>
  $<TARGET_OBJECTS:npunlock_graphinfer_objects>
)
add_library(npunlock::native ALIAS npunlock_native)
set_target_properties(npunlock_native PROPERTIES OUTPUT_NAME npunlock EXPORT_NAME native)
target_include_directories(npunlock_native PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(npunlock_native INTERFACE NPUNLOCK_SHARED)

foreach(component npunlock_common shavecc ir2blob patchblob graphinfer)
  add_library(${component} INTERFACE)
  target_include_directories(${component} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries(${component} INTERFACE npunlock_native)
endforeach()
set_target_properties(npunlock_common PROPERTIES EXPORT_NAME common)
add_library(npunlock::common ALIAS npunlock_common)
add_library(npunlock::shavecc ALIAS shavecc)
add_library(npunlock::ir2blob ALIAS ir2blob)
add_library(npunlock::patchblob ALIAS patchblob)
add_library(npunlock::graphinfer ALIAS graphinfer)

add_executable(npurun src/npurun/main.c)
target_include_directories(npurun PRIVATE src src/common src/npurun)
target_link_libraries(npurun
  PRIVATE shavecc ir2blob patchblob graphinfer npunlock_common $<BUILD_INTERFACE:npunlock_warnings>
)

set(NPUNLOCK_FORMAT_SOURCES
  include/npunlock/api.h
  include/npunlock/buffer.h
  include/npunlock/error.h
  include/npunlock/graphinfer.h
  include/npunlock/ir2blob.h
  include/npunlock/npu3720_kernel.h
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
  src/common/worker_process.h
  src/common/worker_process_win32.c
  src/common/worker_protocol.h
  src/ir2blob/ir2blob.c
  src/ir2blob/ir_worker.h
  src/ir2blob/ir_worker_win32.c
  src/graphinfer/graphinfer.c
  src/graphinfer/infer_worker.h
  src/graphinfer/infer_worker_win32.c
  src/graphinfer/session_win32.c
  src/npurun/main.c
  src/patchblob/patchblob.c
  src/patchblob/graph_blob.c
  src/patchblob/graph_blob.h
  src/shavecc/shavecc.c
  src/shavecc/movi_stage.h
  src/shavecc/movi_stage_win32.c
  src/workers/movi_protocol.h
  src/workers/movi_worker.c
  src/workers/worker.c
  src/workers/worker_entry.h
  src/workers/worker_io.c
  src/workers/worker_io.h
  src/workers/ir_protocol.h
  src/workers/ir_worker.c
  src/workers/level_zero_min.h
  src/workers/infer_protocol.h
  src/workers/infer_worker.c
  tests/package/smoke.c
  tests/integration/test_shavecc.c
  tests/integration/test_ir2blob.c
  tests/integration/test_graphinfer.c
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
  add_dependencies(npunlock_test_api npunlock_worker)
  target_link_libraries(
    npunlock_test_api PRIVATE shavecc ir2blob patchblob graphinfer npunlock_common npunlock_warnings
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

  add_executable(npunlock_test_patchblob_integration tests/integration/test_patchblob.c)
  target_link_libraries(
    npunlock_test_patchblob_integration PRIVATE patchblob npunlock_common npunlock_warnings
  )
  add_test(NAME patchblob_shared_add1 COMMAND npunlock_test_patchblob_integration)
  set_tests_properties(
    patchblob_shared_add1 PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  if(NPUNLOCK_ENABLE_MOVITOOLS_TESTS)
    add_executable(npunlock_test_shavecc_integration tests/integration/test_shavecc.c)
    target_link_libraries(
      npunlock_test_shavecc_integration PRIVATE shavecc npunlock_common npunlock_warnings
    )
    add_dependencies(npunlock_test_shavecc_integration npunlock_worker)
    add_test(NAME movitools_add1 COMMAND npunlock_test_shavecc_integration)
    set_tests_properties(
      movitools_add1 PROPERTIES TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()

  if(NPUNLOCK_ENABLE_NPU_TESTS)
    add_executable(npunlock_test_ir2blob_integration tests/integration/test_ir2blob.c)
    target_link_libraries(
      npunlock_test_ir2blob_integration PRIVATE ir2blob npunlock_common npunlock_warnings
    )
    add_dependencies(npunlock_test_ir2blob_integration npunlock_worker)
    add_test(NAME npu_ir2blob_abs COMMAND npunlock_test_ir2blob_integration)
    set_tests_properties(
      npu_ir2blob_abs PROPERTIES TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()

  if(NPUNLOCK_ENABLE_GRAPHINFER_TESTS)
    add_executable(npunlock_test_graphinfer_integration tests/integration/test_graphinfer.c)
    target_link_libraries(
      npunlock_test_graphinfer_integration PRIVATE graphinfer npunlock_common npunlock_warnings
    )
    add_dependencies(npunlock_test_graphinfer_integration npunlock_worker)
    add_test(NAME npu_graphinfer COMMAND npunlock_test_graphinfer_integration)
    set_tests_properties(
      npu_graphinfer PROPERTIES TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
endif()

install(TARGETS npunlock_native
  EXPORT npunlockTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT PythonRuntime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT PythonRuntime
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)
install(TARGETS npunlock_worker
  EXPORT npunlockTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT PythonRuntime
)
install(TARGETS npurun
  EXPORT npunlockTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
)
install(TARGETS npunlock_common shavecc ir2blob patchblob graphinfer
  EXPORT npunlockTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development)
install(EXPORT npunlockTargets
  NAMESPACE npunlock::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/npunlock
  COMPONENT Development
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
  COMPONENT Development
)
