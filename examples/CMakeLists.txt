add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    set(absl_DIR "$ENV{CONDA_PREFIX}/lib/cmake/absl")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama_infer
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif ()
set_target_properties(llama_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# examples executables
add_executable(examples main.cpp)
target_link_libraries(examples PRIVATE llama gflags glog::glog)
if (LLAMA3_SUPPORT)
    target_link_libraries(examples
        PRIVATE
        absl::base
        absl::strings
        re2::re2
        nlohmann_json::nlohmann_json
    )
endif()
set_target_properties(examples PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
