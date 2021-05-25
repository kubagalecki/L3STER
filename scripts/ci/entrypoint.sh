#!/bin/bash

. /spack/share/spack/setup-env.sh
spack load eigen catch2 intel-tbb trilinos
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
  -DL3STER_ENABLE_TESTS=ON \
  -DL3STER_ENABLE_VERBOSITY=ON \
  .. || exit 1
cmake --build . -- -j || exit 1
ctest --output-on-failure --repeat until-pass:2 --timeout 600 || exit 1
if [ "$REPORT_COVERAGE" != "" ]; then
  chmod +x generate_coverage_report.sh
  ./generate_coverage_report.sh || exit 1
  curl -s https://codecov.io/bash >codecov.sh || exit 1
  chmod +x codecov.sh
  ./codecov.sh -Z -X gcov || exit 1
fi
