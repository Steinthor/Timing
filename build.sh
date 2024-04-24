#!/bin/bash

set -e

nthreads=$(nproc --all)

if [ ! -d "build" ]; then
  mkdir build
fi
cd build || exit
cmake .. -DCMAKE_BUILD_TYPE="Release" -GNinja
ninja -j "$nthreads"
sudo ninja install

cd ..