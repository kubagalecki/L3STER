# Toolchain for UBSan using gcc
set( CMAKE_CXX_COMPILER mpic++ )
set( CMAKE_CXX_FLAGS_INIT "-fsanitize=undefined" )