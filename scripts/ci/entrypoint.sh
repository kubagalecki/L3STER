#!/bin/sh -l

echo "Loading Trilinos from spack..."
. /spack/share/spack/setup-env.sh
spack load trilinos

echo "Building L3STER tests..."
mkdir build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Debug -DL3STER_ENABLE_TESTS=ON .. || exit
cmake --build . || exit

echo "Running L3STER tests..."
ctest && TEST_STATUS=true || TEST_STATUS=false
$TEST_STATUS || echo "Tests failed. Delaying error until after code coverage is reported..."

echo "Generating code coverage report and uploading to Codecov..."
gcovr -x -r .. -e ../tests -o report.xml
curl -s https://codecov.io/bash | bash -s -- -c -f report.xml -t "$INPUT_CODECOV_TOKEN"
$TEST_STATUS && echo "All tests passed. Finishing..." && exit || \
  echo "Some tests failed. Error wil now be reported..." && exit 1
