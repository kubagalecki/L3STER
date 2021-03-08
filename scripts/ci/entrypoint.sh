#!/bin/sh -l

mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
  -DL3STER_ENABLE_TESTS=ON \
  .. || exit 1
cmake --build . -- -j2 || exit 1
ctest --output-on-failure --repeat until-pass:2 --timeout 900 || exit 1 # allow 1 rerun

if [ $REPORT_COVERAGE = true ]; then
  gcovr -x -r .. -e ../tests -o report.xml
  curl -s https://codecov.io/bash | bash -s -- -c -f report.xml -t "$INPUT_CODECOV_TOKEN"
fi