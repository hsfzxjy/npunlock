cmake_minimum_required(VERSION 3.24)

project(npunlock_package_smoke LANGUAGES C)

find_package(npunlock CONFIG REQUIRED)

add_executable(npunlock_package_smoke smoke.c)
target_link_libraries(
  npunlock_package_smoke
  PRIVATE npunlock::shavecc npunlock::ir2blob npunlock::patchblob npunlock::graphinfer
)
