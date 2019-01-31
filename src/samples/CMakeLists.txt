cmake_minimum_required(VERSION 2.8)
project(snn_samples)

include_directories(.)
include_directories(..)

add_executable(signal signal.c)
target_link_libraries(signal snn)

add_executable(xor xor.c)
target_link_libraries(xor snn)

add_executable(444 444.c)
target_link_libraries(444 snn)
