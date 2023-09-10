#!/bin/bash

cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
