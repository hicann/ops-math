cmake_minimum_required(VERSION 3.16.0)
project(reduce_mean_with_count_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# =============================================================================
# Build Google Test with OLD ABI
# =============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# =============================================================================
# Common UT Framework Sources
# =============================================================================
set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# =============================================================================
# Common Include Directories (x86_64)
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/base
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/graph
)

# =============================================================================
# Common Library Directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64
)

# =============================================================================
# Compiler Options
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Add Subdirectories (only op_host for now)
# =============================================================================
add_subdirectory(op_host)
