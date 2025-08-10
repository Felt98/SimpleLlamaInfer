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
# 确保LLAMA3_SUPPORT默认为ON
if(NOT DEFINED LLAMA3_SUPPORT)
  set(LLAMA3_SUPPORT ON)
endif()

option(LLAMA3_SUPPORT "Enable LLAMA3 support" ${LLAMA3_SUPPORT})
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()

option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()
# if(USE_CPM)
#  # Use CPM to manage dependencies
#  include(cmake/CPM.cmake)
 
#  # 首先尝试找到系统已安装的包，如果找不到再用CPM下载
#  find_package(GTest QUIET)
#  if(NOT GTest_FOUND)
#    CPMAddPackage(
#      NAME GTest
#      GITHUB_REPOSITORY google/googletest
#      VERSION 1.15.0
#    )
#  endif()
 
#  find_package(glog QUIET)
#  if(NOT glog_FOUND)
#    CPMAddPackage(
#      NAME glog
#      GITHUB_REPOSITORY google/glog
#      VERSION 0.7.1
#      OPTIONS "BUILD_TESTING Off"
#    )
#  endif()
 
#  find_package(Armadillo QUIET)
#  if(NOT Armadillo_FOUND)
#    CPMAddPackage(
#      NAME Armadillo
#      GITLAB_REPOSITORY conradsnicta/armadillo-code
#      GIT_TAG 14.0.1
#    )
#  endif()
 
#  find_package(sentencepiece QUIET)
#  if(NOT sentencepiece_FOUND)
#    CPMAddPackage(
#      NAME sentencepiece
#      GITHUB_REPOSITORY google/sentencepiece
#      VERSION 0.2.0
#    )
#  endif()
 
#  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
#    # 优先使用conda环境中的abseil
#    find_package(absl QUIET)
#    if(NOT absl_FOUND)
#      CPMAddPackage(
#        NAME absl
#        GITHUB_REPOSITORY abseil/abseil-cpp
#        GIT_TAG 20250127.0  # 更新到2025年版本
#        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
#      )
#    endif()
   
#    find_package(re2 QUIET)
#    if(NOT re2_FOUND)
#      CPMAddPackage(
#        NAME re2
#        GITHUB_REPOSITORY google/re2
#        GIT_TAG 2024-11-01  # 更新到较新版本
#      )
#    endif()
   
#    find_package(nlohmann_json QUIET)
#    if(NOT nlohmann_json_FOUND)
#      CPMAddPackage(
#        NAME nlohmann_json
#        GITHUB_REPOSITORY nlohmann/json
#        VERSION 3.11.3
#      )
#    endif()
#  endif()
# endif()

# ---------------------------------
find_package(GTest REQUIRED)
# find_package(glog REQUIRED)
# if(glog_FOUND)
#     message(STATUS "Found glog version: ${glog_VERSION}")
#     message(STATUS "glog include dirs: ${glog_INCLUDE_DIRS}")
#     message(STATUS "glog libraries: ${glog_LIBRARIES}")
# else()
#     message(FATAL_ERROR "glog not found!")
# endif()
set(gflags_DIR "$ENV{CONDA_PREFIX}/lib/cmake/gflags")
find_package(gflags REQUIRED)
set(glog_DIR "$ENV{CONDA_PREFIX}/lib/cmake/glog")
find_package(glog REQUIRED)

# 详细诊断
message(STATUS "=== GLOG DEBUG INFO ===")
get_target_property(glog_type glog::glog TYPE)
message(STATUS "glog target type: ${glog_type}")

get_target_property(glog_imported_location glog::glog IMPORTED_LOCATION)
message(STATUS "IMPORTED_LOCATION: ${glog_imported_location}")

get_target_property(glog_imported_location_release glog::glog IMPORTED_LOCATION_RELEASE)
message(STATUS "IMPORTED_LOCATION_RELEASE: ${glog_imported_location_release}")

get_target_property(glog_imported_location_debug glog::glog IMPORTED_LOCATION_DEBUG)
message(STATUS "IMPORTED_LOCATION_DEBUG: ${glog_imported_location_debug}")

get_target_property(glog_interface_include glog::glog INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "INTERFACE_INCLUDE_DIRECTORIES: ${glog_interface_include}")

get_target_property(glog_interface_link_libs glog::glog INTERFACE_LINK_LIBRARIES)
message(STATUS "INTERFACE_LINK_LIBRARIES: ${glog_interface_link_libs}")
message(STATUS "=== END GLOG DEBUG ===")

find_package(Armadillo REQUIRED)

# ======添加这些行======
# if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
#     set(absl_DIR "$ENV{CONDA_PREFIX}/lib/cmake/absl")  # ✅ 显式设置路径
#     find_package(absl REQUIRED)
#     find_package(absl REQUIRED)
#     find_package(re2 REQUIRED)
#     find_package(nlohmann_json REQUIRED)
# endif()
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
endif()
# ========================

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
# target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
# ========================
# if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
#     target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo 
#                          absl::strings absl::base absl::str_format absl::synchronization 
#                          absl::cord absl::hash absl::flat_hash_map absl::flat_hash_set
#                          re2::re2 nlohmann_json::nlohmann_json)
# else()
#     target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
# endif()
# ========================

target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/infer/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/infer)
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
enable_testing()
add_subdirectory(test)
add_subdirectory(demo)
