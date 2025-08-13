
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