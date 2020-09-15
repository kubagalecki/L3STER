#!/bin/sh -l

echo "Starting CI build..."
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
echo "Running tests..."
ctest && TEST_STATUS=true || TEST_STATUS=false
$TEST_STATUS || echo "Tests failed. Delaying error until after code coverage is reported"
echo "Generating code coverage report and uploading to Codecov..."
gcovr -x -r .. -e ../tests -o report.xml
curl -s https://codecov.io/bash | bash -s -- -c -f report.xml -t $INPUT_CODECOV_TOKEN
$TEST_STATUS && echo "All tests passed. Good job! Finishing..." && exit 0 || echo "Some tests failed. I'm going to crash now. Goodbye." && exit 1
