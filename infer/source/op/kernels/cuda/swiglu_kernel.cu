#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
#include "base/cuda_macros.h"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  
  // 使用float4向量化加载，但保持原有的共享内存结构
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  // 向量化加载到共享内存（每4个线程协作加载一个float4）
  const int float4_idx = idx / 4;
  const int lane = idx % 4;
  
  if (float4_idx * 4 + 3 < size) {
    // 完整的float4加载
    float4 in1_vec = FLOAT4_LOAD(in1, float4_idx);
    float4 in2_vec = FLOAT4_LOAD(in2, float4_idx);
    
    // 每个线程负责存储对应的float值
    if (lane == 0) {
      smem1[tid] = in1_vec.x;
      smem2[tid] = in2_vec.x;
    } else if (lane == 1) {
      smem1[tid] = in1_vec.y;
      smem2[tid] = in2_vec.y;
    } else if (lane == 2) {
      smem1[tid] = in1_vec.z;
      smem2[tid] = in2_vec.z;
    } else {
      smem1[tid] = in1_vec.w;
      smem2[tid] = in2_vec.w;
    }
  } else {
    // 处理边界情况，使用标量加载
    smem1[tid] = in1[idx];
    smem2[tid] = in2[idx];
  }
  __syncthreads();

  // 原有的SwiGLU计算逻辑
  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel