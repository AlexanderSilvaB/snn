cmake_minimum_required(VERSION 2.8)
project(snn)

file(GLOB_RECURSE SRC 
    "*.c")

include_directories(.)

add_library(snn ${SRC})
target_link_libraries(snn m)