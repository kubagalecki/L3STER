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
  curl https://keybase.io/codecovsecurity/pgp_keys.asc | gpg --import || exit 1
  curl -Os https://uploader.codecov.io/latest/codecov-linux || exit 1
  curl -Os https://uploader.codecov.io/latest/codecov-linux.SHA256SUM || exit 1
  curl -Os https://uploader.codecov.io/latest/codecov-linux.SHA256SUM.sig || exit 1
  gpg --verify codecov-linux.SHA256SUM.sig codecov-linux.SHA256SUM || exit 1
  sha1sum -a 256 -c codecov-linux.SHA256SUM || exit 1
  chmod +x codecov-linux
  ./codecov-linux -Z -X gcov || exit 1
fi
