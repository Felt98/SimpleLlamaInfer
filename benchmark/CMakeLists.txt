# Benchmark executables
add_executable(performance_test performance_test.cpp)
target_link_libraries(performance_test PRIVATE llama gflags glog::glog)

add_executable(latency_benchmark latency_benchmark.cpp)
target_link_libraries(latency_benchmark PRIVATE llama gflags glog::glog)

add_executable(memory_benchmark memory_benchmark.cpp)
target_link_libraries(memory_benchmark PRIVATE llama gflags glog::glog)

add_executable(accuracy_benchmark accuracy_benchmark.cpp)
target_link_libraries(accuracy_benchmark PRIVATE llama gflags glog::glog)

if (LLAMA3_SUPPORT)
    target_link_libraries(performance_test
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
    target_link_libraries(latency_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
    target_link_libraries(memory_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
    target_link_libraries(accuracy_benchmark
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()

# Include benchmark utils
target_include_directories(performance_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(latency_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(memory_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(accuracy_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(performance_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(latency_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(memory_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(accuracy_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
