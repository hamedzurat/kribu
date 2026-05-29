cmake_minimum_required(VERSION 3.24)
project(kribu LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure compiler launcher (ccache) if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimizations and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-c2y-extensions
    -O3
    -march=native
    -mtune=native
    -flto
    -fomit-frame-pointer)
  add_link_options(-flto)
endif()

# Setup CPM
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION
    "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake")
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# We only build the engine and tests directly with this CMake project. The
# Python bindings could be built here, or via pip/uv with scikit-build. For
# simplicity during development, we include them here.

add_subdirectory(engine)
add_subdirectory(bindings)
