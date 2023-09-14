#!/bin/bash

cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
