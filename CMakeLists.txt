CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(L3STER)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE DEBUG)

if (UNIX)
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
    add_compile_options(/Wall /experimental:newLambdaProcessor)
endif()

# set up clang-tidy
# set(CMAKE_CXX_CLANG_TIDY 
#   clang-tidy;
#   -header-filter=.*\.hpp;
#   -checks=-*,performance-*;)

# EIGEN
include_directories(${CMAKE_SOURCE_DIR}/eigen/)

# SRC
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(main
        ${CMAKE_SOURCE_DIR}/main.cpp)
