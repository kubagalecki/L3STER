CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

PROJECT(L3STER)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

#add_compile_options(-Wall -Wextra -Wpedantic)

# set up clang-tidy
# set(CMAKE_CXX_CLANG_TIDY 
#   clang-tidy;
#   -header-filter=.*\.hpp;
#   -checks=-*,performance-*;)

# PETSC
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_definitions(${PETSC_DEFINITIONS})

# EIGEN
include_directories(${CMAKE_SOURCE_DIR}/eigen-git-mirror/)

# SRC
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(main
${CMAKE_SOURCE_DIR}/main.cpp
${CMAKE_SOURCE_DIR}/src/mesh/init_elements.cpp
)

target_link_libraries(main ${PETSC_LIBRARIES})
