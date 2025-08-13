#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>
#include "base/cuda_macros.h"
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"

namespace kernel {
// CUDA常量定义
// Warp Reduce Sum
template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// weight: [K×M], input: [M], output: [K]
// 每个线程负责4个元素，一个warp覆盖128个元素
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~K/4
  int lane = tx % WARP_SIZE;    // 0~31
  int k_idx = blockDim.y * bx + ty; // (0~K/4) * 4 + (0~3)

  if (k_idx < K) {
    float sum = 0.0f;
    // process 4*WARP_SIZE elements per warp.
    int NUM_WARPS = (((M + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int m = (w * WARP_SIZE + lane) * 4;
      if (m + 3 < M) {
        // 使用float4向量化加载
        float4 reg_input = FLOAT4_LOAD(input, m / 4);
        float4 reg_weight = FLOAT4_LOAD(weight + k_idx * M, m / 4);
        sum += (reg_weight.x * reg_input.x + reg_weight.y * reg_input.y + 
                reg_weight.z * reg_input.z + reg_weight.w * reg_input.w);
      } else {
        // 处理边界情况
        for (int i = 0; i < 4 && m + i < M; ++i) {
          sum += input[m + i] * weight[k_idx * M + m + i];
        }
      }
    }
    
    // Warp内归约求和
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    
    // 写回结果
    if (lane == 0) {
      output[k_idx] = sum;
    }
  }
}

// 优化的INT8 matmul kernel: 使用warp归约和向量化
// weight: [K×M] (int8), input: [M] (fp32), output: [K] (fp32)
// 每个线程负责4个元素，一个warp覆盖128个元素
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~K/4
  int lane = tx % WARP_SIZE;    // 0~31
  int k_idx = blockDim.y * bx + ty; // (0~K/4) * 4 + (0~3)

  if (k_idx < K) {
    float sum = 0.0f;
    // process 4*WARP_SIZE elements per warp.
    int NUM_WARPS = (((M + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int m = (w * WARP_SIZE + lane) * 4;
      if (m + 3 < M) {
        // 使用float4向量化加载input
        float4 reg_input = FLOAT4_LOAD(input, m / 4);
        
        // 处理int8权重 - 需要4个int8值
        const int8_t* weight_ptr = weight + k_idx * M + m;
        int8_t weight_vals[4];
        weight_vals[0] = weight_ptr[0];
        weight_vals[1] = weight_ptr[1];
        weight_vals[2] = weight_ptr[2];
        weight_vals[3] = weight_ptr[3];
        
        // 计算对应的scale索引
        const int weight_idx_base = k_idx * M + m;
        for (int i = 0; i < 4; ++i) {
          const int weight_idx = weight_idx_base + i;
          const int group_idx = weight_idx / group_size;
          sum += reg_input.x * scales[group_idx] * static_cast<float>(weight_vals[i]);
        }
      } else {
        // 处理边界情况
        for (int i = 0; i < 4 && m + i < M; ++i) {
          const int weight_idx = k_idx * M + m + i;
          const int group_idx = weight_idx / group_size;
          sum += input[m + i] * scales[group_idx] * static_cast<float>(weight[k_idx * M + m + i]);
        }
      }
    }
    
    // Warp内归约求和
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    
    // 写回结果
    if (lane == 0) {
      output[k_idx] = sum;
    }
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK_EQ(M, input.get_dim(0));
  
  if (config && config->stream) {
    dim3 block_dim(32, 4);  // 32×4=128个线程
    dim3 grid_dim((K + 4 - 1) / 4);  // 每个block处理4个输出
    matmul_kernel_cu_fp32<<<grid_dim, block_dim, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    dim3 block_dim(32, 4);  // 32×4=128个线程
    dim3 grid_dim((K + 4 - 1) / 4);  // 每个block处理4个输出
    matmul_kernel_cu_fp32<<<grid_dim, block_dim>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    dim3 block_dim(32, 4);  // 32×4=128个线程
    dim3 grid_dim((K + 4 - 1) / 4);  // 每个block处理4个输出
    matmul_kernel_cu_fp32int8<<<grid_dim, block_dim, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    dim3 block_dim(32, 4);  // 32×4=128个线程
    dim3 grid_dim((K + 4 - 1) / 4);  // 每个block处理4个输出
    matmul_kernel_cu_fp32int8<<<grid_dim, block_dim>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}

// cuBLAS实现版本
namespace {
// 全局cuBLAS句柄
cublasHandle_t g_cublas_handle = nullptr;

void init_cublas() {
  if (g_cublas_handle == nullptr) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "cuBLAS create failed: " << status;
    }
  }
}

void cleanup_cublas() {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
}
} // anonymous namespace

void matmul_kernel_cublas(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  
  const int32_t K = weight.get_dim(0);  // 权重矩阵行数
  const int32_t M = weight.get_dim(1);  // 权重矩阵列数
  CHECK_EQ(M, input.get_dim(0));
  CHECK_EQ(K, output.size());

  // 初始化cuBLAS
  init_cublas();
  if (g_cublas_handle == nullptr) {
    LOG(ERROR) << "Failed to initialize cuBLAS handle";
    return;
  }

  // 设置CUDA流
  if (config && config->stream) {
    cublasStatus_t stream_status = cublasSetStream(g_cublas_handle, config->stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "Failed to set cuBLAS stream: " << stream_status;
    }
  }

  // 执行GEMV: y = scale * A^T * x + 0 * y
  // 由于cuBLAS使用列优先存储，我们需要转置权重矩阵
  // A = weight [K×M], x = input [M], y = output [K]
  float alpha = scale;
  float beta = 0.0f;
  
  cublasStatus_t status = cublasSgemv(
      g_cublas_handle,
      CUBLAS_OP_T,                          // 转置权重矩阵
      M,                                    // 转置后的矩阵行数
      K,                                    // 转置后的矩阵列数
      &alpha,                               // alpha缩放因子
      weight.ptr<float>(),                  // 权重矩阵
      M,                                    // leading dimension (原始矩阵的列数)
      input.ptr<float>(),                   // 输入向量
      1,                                    // 步长
      &beta,                                // beta缩放因子
      const_cast<float*>(output.ptr<float>()), // 输出向量
      1                                     // 步长
  );
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "cuBLAS GEMV failed with status: " << status;
  }
}

}  // namespace kernel