include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  cpuinfo
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/chengcli/cpuinfo/archive/refs/tags/v240531.tar.gz)

set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cpuinfo)

include_directories(${cpuinfo_SOURCE_DIR})
