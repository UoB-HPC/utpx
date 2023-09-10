cmake_minimum_required(VERSION 3.14)
project(utpx)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        elfio
        URL https://github.com/serge1/ELFIO/releases/download/Release_3.12/elfio-3.12.tar.gz
        URL_HASH SHA256=caf49f3bf55a9c99c98ebea4b05c79281875783802e892729eea0415505f68c4
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        URL_HASH SHA256=665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
)
FetchContent_MakeAvailable(elfio)
FetchContent_MakeAvailable(json)


add_library(utpx SHARED
        utpx.cpp
        intercept_kernel.cpp
        intercept_memory.cpp
        hsaco.cpp
)
target_link_libraries(utpx PRIVATE elfio::elfio)
target_include_directories(utpx PRIVATE ${json_SOURCE_DIR})
target_compile_options(utpx PRIVATE
        "-Wall"
        "-Wno-unused-variable"
        "-ffunction-sections"
)