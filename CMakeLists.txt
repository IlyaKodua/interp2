cmake_minimum_required(VERSION 3.10)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)

project(int2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(int2 example.cu interp2.cuh)

set_target_properties(
        int2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
