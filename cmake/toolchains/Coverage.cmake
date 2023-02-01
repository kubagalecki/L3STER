# Toolchain for code coverage using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-fprofile-arcs -ftest-coverage -fPIC -lgcov --coverage -p -Wno-volatile -Wno-unused-parameter -Wno-deprecated-enum-enum-conversion -Wno-deprecated-declarations" )