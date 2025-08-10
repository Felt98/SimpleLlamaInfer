
add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    set(absl_DIR "$ENV{CONDA_PREFIX}/lib/cmake/absl")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    # target_link_libraries(llama_infer absl::base re2::re2 nlohmann_json::nlohmann_json)
    target_link_libraries(llama_infer
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif ()
set_target_properties(llama_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 移除重复的 find_package，因为主 CMakeLists.txt 已经处理了
# if (LLAMA3_SUPPORT)
#     message(STATUS "LINK LLAMA3 SUPPORT")
#     # 不需要额外链接，因为 llama 库已经包含了这些依赖
# endif ()
# set_target_properties(llama_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ========================

if (QWEN2_SUPPORT)
    message(STATUS "LINK QWEN2 SUPPORT")
    add_executable(qwen_infer main_qwen.cpp)
    target_link_directories(qwen_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(qwen_infer PRIVATE llama glog::glog)
    target_link_libraries(qwen_infer
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
    set_target_properties(qwen_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
elseif (QWEN3_SUPPORT)
    message(STATUS "LINK QWEN3 SUPPORT")
    add_executable(qwen3_infer main_qwen3.cpp)
    target_link_directories(qwen3_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(qwen3_infer PRIVATE llama glog::glog)
    target_link_libraries(qwen3_infer
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
    set_target_properties(qwen3_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# Demo executables
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE llama gflags glog::glog)

# Performance benchmark executables
add_executable(performance_test benchmark/performance_test.cpp)
target_link_libraries(performance_test  PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    target_link_libraries(performance_test
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()

add_executable(latency_benchmark benchmark/latency_benchmark.cpp)
target_link_libraries(latency_benchmark PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    target_link_libraries(latency_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()

add_executable(memory_benchmark benchmark/memory_benchmark.cpp)  
target_link_libraries(memory_benchmark PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    target_link_libraries(memory_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()

add_executable(accuracy_benchmark benchmark/accuracy_benchmark.cpp)
target_link_libraries(accuracy_benchmark PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    target_link_libraries(accuracy_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()

# Include benchmark utils
target_include_directories(performance_test PRIVATE benchmark)
target_include_directories(latency_benchmark PRIVATE benchmark)
target_include_directories(memory_benchmark PRIVATE benchmark)
target_include_directories(accuracy_benchmark PRIVATE benchmark)