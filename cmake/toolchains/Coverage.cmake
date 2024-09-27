# Toolchain for code coverage using gcc
set( CMAKE_CXX_COMPILER mpicxx )
set( CMAKE_CXX_FLAGS_INIT "-march=native -mtune=native -fprofile-arcs -fprofile-abs-path -ftest-coverage -fPIC -lgcov --coverage -p" )