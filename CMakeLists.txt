CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(L3STER)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

#add_compile_options(-Wall -Wextra -Wpedantic)

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
${CMAKE_SOURCE_DIR}/main.cpp
${CMAKE_SOURCE_DIR}/src/mesh/init_elements.cpp
)
