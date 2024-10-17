# Toolchain for static analysis using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-march=native -mtune=native -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -Wno-interference-size" )
