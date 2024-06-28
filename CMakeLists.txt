cmake_minimum_required (VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

project (run_weno)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Default Build Type = Release")
  message(STATUS "")
else()
  message(STATUS "Build Type = ${CMAKE_BUILD_TYPE}")
  message(STATUS "")
endif()

# Add debugging flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# load all modules
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)

# load all macros
file(GLOB _macro_files "${CMAKE_SOURCE_DIR}/cmake/macros/*.cmake")
foreach(_file ${_macro_files})
  message(STATUS "Include ${_file}")
  include(${_file})
endforeach()

find_package(Torch REQUIRED)

find_program(NC_CONFIG nc-config)
if (NOT NC_CONFIG)
    message(FATAL_ERROR "nc-config not found. Please ensure NetCDF is installed.")
endif()

execute_process(COMMAND ${NC_CONFIG} --cflags OUTPUT_VARIABLE NETCDF_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${NC_CONFIG} --libs OUTPUT_VARIABLE NETCDF_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "-I([^ ]+)" _ ${NETCDF_CFLAGS})
set(NETCDF_INCLUDE_DIR ${CMAKE_MATCH_1})

string(REGEX MATCH "-L([^ ]+)" _ ${NETCDF_LIBS})
set(NETCDF_LIBRARY_DIR ${CMAKE_MATCH_1})
string(REGEX REPLACE "-L[^ ]+ " "" NETCDF_LIBS ${NETCDF_LIBS})
string(REGEX REPLACE "-l" "" NETCDF_LIBS ${NETCDF_LIBS})

message(STATUS "NetCDF include flags: ${NETCDF_INCLUDE_DIR}")
message(STATUS "NetCDF libraries: ${NETCDF_LIBRARY_DIR}")
message(STATUS "NetCDF libraries: ${NETCDF_LIBS}")

link_directories(${NETCDF_LIBRARY_DIR})

add_executable(run_weno_1d
  main_1d.cpp
  weno5_advection.cpp
  netcdf_output.cpp
)

add_executable(run_weno_2d
  main_2d.cpp
  weno5_advection.cpp
  netcdf_output.cpp
)

target_include_directories(run_weno_1d
  SYSTEM PUBLIC ${TORCH_API_INCLUDE_DIR}
  SYSTEM PUBLIC ${TORCH_INCLUDE_DIR}
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
  SYSTEM PUBLIC ${NETCDF_INCLUDE_DIR}
)

target_link_libraries(run_weno_1d
  ${TORCH_LIBRARY}
  ${TORCH_CPU_LIBRARY}
  ${C10_LIBRARY}
  ${NETCDF_LIBS}
)

target_include_directories(run_weno_2d
  SYSTEM PUBLIC ${TORCH_API_INCLUDE_DIR}
  SYSTEM PUBLIC ${TORCH_INCLUDE_DIR}
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
  SYSTEM PUBLIC ${NETCDF_INCLUDE_DIR}
)

target_link_libraries(run_weno_2d
  ${TORCH_LIBRARY}
  ${TORCH_CPU_LIBRARY}
  ${C10_LIBRARY}
  ${NETCDF_LIBS}
)