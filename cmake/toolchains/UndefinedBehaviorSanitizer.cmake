# Toolchain for UBSan using gcc-10
set( CMAKE_CXX_COMPILER mpic++ )
set( CMAKE_CXX_FLAGS_INIT "-fsanitize=undefined" )