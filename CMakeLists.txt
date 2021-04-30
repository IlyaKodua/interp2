cmake_minimum_required(VERSION 3.19)
project(int2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(int2 main.cu interp2.cuh)

set_target_properties(
        int2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)