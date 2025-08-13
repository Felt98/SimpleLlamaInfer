include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../infer/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)


if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()
# 修正：这里应该是 test_llm 而不是 llama
# if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
#     message(STATUS "LINK LLAMA3 SUPPORT")
#     # 不需要重复 find_package，直接链接即可
# endif ()
# ========================

target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 设置 RPATH，让可执行文件自动找到 conda 环境的库
set_target_properties(test_llm PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib:${PROJECT_SOURCE_DIR}/lib"
)

# 设置测试环境变量，确保能找到 conda 环境的库
set_property(TARGET test_llm PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=$ENV{CONDA_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}")
# ✅ 注册测试
include(GoogleTest)
