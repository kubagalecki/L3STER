# Toolchain for TSan using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-march=native -mtune=native -Wno-interference-size -fsanitize=thread" )