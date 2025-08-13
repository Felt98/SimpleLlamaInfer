cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{CONDA_PREFIX}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
# LLAMA3_SUPPORT默认为ON
if(NOT DEFINED LLAMA3_SUPPORT)
  set(LLAMA3_SUPPORT ON)
endif()

option(LLAMA3_SUPPORT "Enable LLAMA3 support" ${LLAMA3_SUPPORT})
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()

# ---------------------------------
find_package(GTest REQUIRED)

set(gflags_DIR "$ENV{CONDA_PREFIX}/lib/cmake/gflags")
find_package(gflags REQUIRED)
set(glog_DIR "$ENV{CONDA_PREFIX}/lib/cmake/glog")
find_package(glog REQUIRED)

find_package(Armadillo REQUIRED)
if (LLAMA3_SUPPORT)
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
endif()

aux_source_directory(infer/source/tensor/ DIR_TENSOR)
aux_source_directory(infer/source/base/ DIR_BASE)
aux_source_directory(infer/source/op/ DIR_OP)
aux_source_directory(infer/source/model/ DIR_MODEL)
aux_source_directory(infer/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(infer/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(infer/source/op/kernels/ DIR_KERNEL)
aux_source_directory(infer/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama
  sentencepiece
  gflags
  glog::glog
  gtest
  gtest_main
  pthread
  cudart
  armadillo
  absl::base
  absl::strings    
)

target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/infer/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
enable_testing()
add_subdirectory(test)
add_subdirectory(demo)
