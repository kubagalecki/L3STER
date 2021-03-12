#!/bin/sh -l

mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
  -DL3STER_ENABLE_TESTS=ON \
  .. || exit 1
cmake --build . -- -j || exit 1
ctest --output-on-failure --repeat until-pass:2 --timeout 120 || exit 1

if [ $REPORT_COVERAGE = true ]; then
  bash generate_coverage_report.sh || exit 1
  curl -s https://codecov.io/bash | bash -Z -f coverage_report.json -t "$INPUT_CODECOV_TOKEN" || exit 1
fi