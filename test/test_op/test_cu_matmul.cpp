#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cuda/matmul_kernel.cuh"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;
TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          nullptr);

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 3, true, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

// 测试cuBLAS实现的正确性 - 简化版本
TEST(test_matmul_cu, matmul_cublas_correctness) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 创建简单的测试数据 - 和原有测试保持一致
  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  // 初始化输入数据 - 和matmul_linear_course一样的数据
  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  // 初始化权重数据 - 和matmul_linear_course一样的数据
  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  // 复制到CUDA
  tensor::Tensor input_cu = input.clone();
  tensor::Tensor weight_cu = weight.clone();
  input_cu.to_cuda();
  weight_cu.to_cuda();

  // 创建输出tensor
  tensor::Tensor out_cublas(base::DataType::kDataTypeFp32, 3, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  // 使用CPU计算作为参考
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input, weight, out_cpu, 1.f, nullptr);

  // 使用cuBLAS计算 - 不使用stream先
  kernel::matmul_kernel_cublas(input_cu, weight_cu, out_cublas, 1.f, nullptr);

  // 同步等待计算完成
  cudaDeviceSynchronize();

  // 将结果拷贝到CPU进行比较
  out_cublas.to_cpu();

  // 验证cuBLAS结果与CPU结果一致 - 预期结果应该是[0, 3, 6]
  ASSERT_NEAR(out_cublas.index<float>(0), 0.0f, 1e-4f);
  ASSERT_NEAR(out_cublas.index<float>(1), 3.0f, 1e-4f);
  ASSERT_NEAR(out_cublas.index<float>(2), 6.0f, 1e-4f);

  // 也验证与CPU结果的一致性
  for (int i = 0; i < out_cpu.size(); ++i) {
    ASSERT_NEAR(out_cublas.index<float>(i), out_cpu.index<float>(i), 1e-4f)
        << "cuBLAS result differs from CPU at index " << i
        << ": cuBLAS=" << out_cublas.index<float>(i) 
        << ", CPU=" << out_cpu.index<float>(i);
  }
}

// 测试cuBLAS大规模矩阵乘法
TEST(test_matmul_cu, matmul_cublas_large_scale) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 1024;  // 输入向量长度
  const int K = 512;   // 输出向量长度

  // 创建测试数据
  tensor::Tensor input(base::DataType::kDataTypeFp32, M, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, K, M, true, alloc_cpu);

  // 初始化数据
  for (int i = 0; i < M; ++i) {
    input.index<float>(i) = static_cast<float>(i % 100) / 100.0f;
  }

  for (int i = 0; i < K * M; ++i) {
    weight.index<float>(i) = static_cast<float>(i % 50) / 50.0f;
  }

  // 复制到CUDA
  tensor::Tensor input_cu = input.clone();
  tensor::Tensor weight_cu = weight.clone();
  input_cu.to_cuda();
  weight_cu.to_cuda();

  // 创建输出tensor
  tensor::Tensor out_custom(base::DataType::kDataTypeFp32, K, true, alloc_cu);
  tensor::Tensor out_cublas(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 创建CUDA配置
  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;

  // 单次执行测试一致性
  // 使用自定义kernel计算
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cu, weight_cu, out_custom, 1.f, config);

  // 使用cuBLAS计算
  kernel::matmul_kernel_cublas(input_cu, weight_cu, out_cublas, 1.f, config);

  // 同步等待计算完成
  cudaStreamSynchronize(stream);

  // 将结果拷贝到CPU进行比较
  out_custom.to_cpu();
  out_cublas.to_cpu();

  // 更全面的结果验证策略
  float max_relative_error = 0.0f;
  float avg_relative_error = 0.0f;
  int valid_comparisons = 0;
  int large_error_count = 0;
  
  for (int i = 0; i < K; ++i) {
    float custom_result = out_custom.index<float>(i);
    float cublas_result = out_cublas.index<float>(i);
    
    // 跳过接近零的值，避免相对误差被放大
    float abs_max = std::max(std::abs(custom_result), std::abs(cublas_result));
    if (abs_max < 1e-6f) continue;
    
    float relative_error = std::abs(custom_result - cublas_result) / abs_max;
    max_relative_error = std::max(max_relative_error, relative_error);
    avg_relative_error += relative_error;
    valid_comparisons++;
    
    // 统计大误差元素的数量
    if (relative_error > 5e-2f) {
      large_error_count++;
      // 只打印前几个大误差样本，避免日志过多
      if (large_error_count <= 3) {
        LOG(WARNING) << "Large error at index " << i 
                     << ": custom=" << custom_result 
                     << ", cuBLAS=" << cublas_result
                     << ", relative_error=" << relative_error;
      }
    }
  }
  
  if (valid_comparisons > 0) {
    avg_relative_error /= valid_comparisons;
  }
  
  // 统计信息
  LOG(INFO) << "Valid comparisons: " << valid_comparisons << "/" << K;
  LOG(INFO) << "Max relative error: " << max_relative_error;
  LOG(INFO) << "Avg relative error: " << avg_relative_error;
  LOG(INFO) << "Large error count: " << large_error_count;
  
  // 验证标准：
  // 1. 平均相对误差 < 1%
  // 2. 最大相对误差 < 10%  
  // 3. 大误差元素 < 5%
  ASSERT_LT(avg_relative_error, 1e-2f) << "Average relative error too large";
  ASSERT_LT(max_relative_error, 1e-1f) << "Maximum relative error too large";
  ASSERT_LT(static_cast<float>(large_error_count) / valid_comparisons, 0.05f) 
      << "Too many elements with large errors";

  // 清理资源
  cudaStreamDestroy(stream);
  delete config;
}

// 性能比较测试
TEST(test_matmul_cu, matmul_cublas_performance) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 2048;
  const int K = 1024;
  const int num_iterations = 100;

  // 创建测试数据
  tensor::Tensor input(base::DataType::kDataTypeFp32, M, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, K, M, true, alloc_cpu);

  // 随机初始化数据
  for (int i = 0; i < M; ++i) {
    input.index<float>(i) = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < K * M; ++i) {
    weight.index<float>(i) = static_cast<float>(rand()) / RAND_MAX;
  }

  // 复制到CUDA
  tensor::Tensor input_cu = input.clone();
  tensor::Tensor weight_cu = weight.clone();
  input_cu.to_cuda();
  weight_cu.to_cuda();

  // 创建输出tensor
  tensor::Tensor out_custom(base::DataType::kDataTypeFp32, K, true, alloc_cu);
  tensor::Tensor out_cublas(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 创建CUDA配置
  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;

  // 预热
  for (int i = 0; i < 10; ++i) {
    kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cu, weight_cu, out_custom, 1.f, config);
    kernel::matmul_kernel_cublas(input_cu, weight_cu, out_cublas, 1.f, config);
  }
  cudaStreamSynchronize(stream);

  // 测试自定义kernel性能
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cu, weight_cu, out_custom, 1.f, config);
  }
  cudaStreamSynchronize(stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto custom_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 测试cuBLAS性能
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    kernel::matmul_kernel_cublas(input_cu, weight_cu, out_cublas, 1.f, config);
  }
  cudaStreamSynchronize(stream);
  end = std::chrono::high_resolution_clock::now();
  auto cublas_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 输出性能结果
  LOG(INFO) << "Matrix size: " << K << "x" << M;
  LOG(INFO) << "Custom kernel time: " << custom_time / 1000.0f << " ms";
  LOG(INFO) << "cuBLAS time: " << cublas_time / 1000.0f << " ms";
  LOG(INFO) << "Speedup: " << static_cast<float>(custom_time) / cublas_time << "x";

  // 验证最终结果的正确性
  out_custom.to_cpu();
  out_cublas.to_cpu();
  for (int i = 0; i < std::min(10, K); ++i) {
    ASSERT_NEAR(out_cublas.index<float>(i), out_custom.index<float>(i), 20.0f);  // 放宽绝对误差
  }

  // 清理资源
  cudaStreamDestroy(stream);
  delete config;
}

