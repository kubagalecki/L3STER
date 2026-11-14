# Toolchain for UBSan using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-march=native -mtune=native -Wno-interference-size -Wno-stringop-overflow -fsanitize=undefined" )
set( L3STER_ENABLE_MPI_TESTS OFF )