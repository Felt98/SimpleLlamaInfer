#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
// RAII配置cuda流
struct CudaConfig {
  // 流
  cudaStream_t stream = nullptr;

  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  // 初始化
  void init() {
    cudaStreamCreate(&stream);
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
