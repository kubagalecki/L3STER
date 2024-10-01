#!/bin/bash

. /spack/share/spack/setup-env.sh
spack env activate build-env

cd examples || exit 1
mkdir -p build
cd build || exit 1
cmake \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_PATH" \
  .. || exit 1
cmake --build . -- -j "$(grep -c ^processor /proc/cpuinfo)" || exit 1
