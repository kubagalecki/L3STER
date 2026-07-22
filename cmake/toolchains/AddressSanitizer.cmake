# Toolchain for ASan using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-march=native -mtune=native -Wno-interference-size -Wno-array-bounds -Wno-stringop-overflow -fsanitize=address -fno-omit-frame-pointer" )
set( L3STER_ENABLE_MPI_TESTS OFF )