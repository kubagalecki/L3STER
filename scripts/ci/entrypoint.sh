#!/bin/bash

. /spack/share/spack/setup-env.sh
spack load eigen catch2 intel-tbb trilinos
mkdir build
cd build || exit 1
cmake \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_PATH" \
  -DL3STER_ENABLE_TESTS=ON \
  -DL3STER_ENABLE_VERBOSITY=ON \
  .. || exit 1
cmake --build . -- -j || exit 1
ctest --output-on-failure --repeat until-pass:2 --timeout 600 || exit 1
if [ "$REPORT_COVERAGE" != "" ]; then
  chmod +x generate_coverage_report.sh
  ./generate_coverage_report.sh || exit 1
  curl https://keybase.io/codecovsecurity/pgp_keys.asc | gpg --import || exit 1
  curl -Os https://uploader.codecov.io/latest/linux/codecov || exit 1
  curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM || exit 1
  curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM.sig || exit 1
  gpg --verify codecov.SHA256SUM.sig codecov.SHA256SUM || exit 1
  sha256sum -c codecov.SHA256SUM || exit 1
  rm generate_coverage_report.sh codecov.SHA256SUM.sig codecov.SHA256SUM
  chmod +x codecov
  ./codecov -Z -X gcov || exit 1
fi
