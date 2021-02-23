# Toolchain for ASan using gcc-10
set( CMAKE_CXX_COMPILER g++-10 )
set( CMAKE_CXX_FLAGS_INIT "-fsanitize=address" )