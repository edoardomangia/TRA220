cmake_minimum_required(VERSION 3.18)
project(3Dpoisson LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(3Dpoisson
    src/main.cpp
    src/grid3d.cpp
    # src/poisson_coeffs.cpp
    src/poisson_solver.cu
    # src/gen_vti.cpp
)

target_include_directories(3Dpoisson PRIVATE 
    include
    ${CUDAToolkit_INCLUDE_DIRS}
)

