#!/bin/bash

if [ -n "$DEPLOYMENT_TESTS" ]; then
  TEST_DIR=.
else
  TEST_DIR=tests/
fi

. /spack/share/spack/setup-env.sh
spack load eigen catch2 tbb trilinos mpi metis

mkdir -p build
cd build || exit 1
cmake \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_PATH" \
  -DL3STER_ENABLE_COVERAGE="$GATHER_COVERAGE" \
  -DCMAKE_INSTALL_PREFIX="$GITHUB_WORKSPACE/install" \
  -DL3STER_ENABLE_TESTS=ON \
  -DL3STER_ENABLE_VERBOSITY=ON \
  .. || exit 1

if [ -n "$DEPLOYMENT_TESTS" ]; then
  cmake --install . || exit 1
  rm -rf ./*
  cmake \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_PATH" \
    -DCMAKE_INSTALL_PREFIX="$GITHUB_WORKSPACE/install" \
    -DL3STER_ENABLE_VERBOSITY=ON \
    ../tests || exit 1
fi

cmake --build . -- -j "$(grep -c ^processor /proc/cpuinfo)" || exit 1
ctest --output-on-failure --timeout 300 --test-dir $TEST_DIR || exit 1
