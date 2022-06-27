# Toolchain for code coverage using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-fprofile-arcs -ftest-coverage -fPIC -lgcov --coverage -p" )