# Toolchain for static analysis using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-Wno-volatile -Wno-unused-parameter -Wno-deprecated-enum-enum-conversion -Wno-deprecated-declarations -Wno-nonnull" )
