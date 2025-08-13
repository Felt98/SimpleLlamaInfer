#include "add_kernel.cuh"
#include "base/cuda_macros.h"

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  // 向量化处理，每个线程处理4个元素
  int32_t vec_idx = tid * 4;
  
  if (vec_idx + 3 < size) {
    float4 in_val1 = FLOAT4_LOAD(in1, tid);
    float4 in_val2 = FLOAT4_LOAD(in2, tid);
    
    float4 out_val;
    out_val.x = in_val1.x + in_val2.x;
    out_val.y = in_val1.y + in_val2.y;
    out_val.z = in_val1.z + in_val2.z;
    out_val.w = in_val1.w + in_val2.w;

    reinterpret_cast<float4*>(out)[tid] = out_val;
  } else {
    // 处理边界情况
    for (int i = 0; i < 4 && vec_idx + i < size; ++i) {
      out[vec_idx + i] = in1[vec_idx + i] + in2[vec_idx + i];
    }
  }
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  // 每个线程处理4个元素，所以需要的线程数减少到1/4
  int32_t elements_per_thread = 4;
  int32_t block_num = (size + thread_num * elements_per_thread - 1) / (thread_num * elements_per_thread);
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel
