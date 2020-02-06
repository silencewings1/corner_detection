cmake_minimum_required(VERSION 3.0)
project(CornerDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -pthread")
# set(CMAKE_BUILD_TYPE debug)

include_directories(src)
add_subdirectory(src)
