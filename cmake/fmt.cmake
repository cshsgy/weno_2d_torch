include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  fmt 
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip)

FetchContent_MakeAvailable(fmt)

include_directories(${fmt_SOURCE_DIR}/include)
