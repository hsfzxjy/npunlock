cmake_minimum_required(VERSION 3.24)

project(npunlock VERSION 0.1.0 LANGUAGES C)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NPUNLOCK_BUILD_TESTS "Build offline unit tests" ON)
option(NPUNLOCK_ENABLE_NPU_TESTS "Enable tests that require an Intel NPU" OFF)

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

add_library(shavecc src/shavecc/shavecc.c)
add_library(npunlock::shavecc ALIAS shavecc)
target_include_directories(shavecc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(shavecc PRIVATE src/common)
target_link_libraries(shavecc PRIVATE npunlock_common $<BUILD_INTERFACE:npunlock_warnings>)

add_library(ir2blob src/ir2blob/ir2blob.c)
add_library(npunlock::ir2blob ALIAS ir2blob)
target_include_directories(ir2blob PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(ir2blob PRIVATE src/common)
target_link_libraries(ir2blob PRIVATE npunlock_common $<BUILD_INTERFACE:npunlock_warnings>)

add_library(patchblob src/patchblob/patchblob.c)
add_library(npunlock::patchblob ALIAS patchblob)
target_include_directories(patchblob PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(patchblob PRIVATE src/common)
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

add_executable(npurun src/npurun/main.c)
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
  src/common/status.c
  src/ir2blob/ir2blob.c
  src/npurun/main.c
  src/patchblob/patchblob.c
  src/shavecc/shavecc.c
  tests/package/smoke.c
  tests/unit/test_api.c
  tests/unit/test_common.c
  tests/unit/test_elf32.c
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
  target_link_libraries(
    npunlock_test_api PRIVATE shavecc ir2blob patchblob npunlock_common npunlock_warnings
  )
  add_test(NAME api_contract COMMAND npunlock_test_api)

  add_executable(npunlock_test_elf32 tests/unit/test_elf32.c)
  target_include_directories(npunlock_test_elf32 PRIVATE src/common)
  target_link_libraries(npunlock_test_elf32 PRIVATE npunlock_common npunlock_warnings)
  add_test(NAME shave_elf_parser COMMAND npunlock_test_elf32)
endif()

install(TARGETS npunlock_common shavecc ir2blob patchblob npurun
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
