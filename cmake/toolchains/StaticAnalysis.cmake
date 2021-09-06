# Toolchain for static analysis using gcc-10
set( CMAKE_CXX_COMPILER mpic++ )
set( CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic -Wshadow -Werror" )
# set( CMAKE_CXX_CLANG_TIDY clang-tidy ) clang-tidy disabled until C++20 is fully supported (currently lots of false positives)