#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
// 贪心采样器
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));  // 获取最大值的索引
    return next;
  } else {
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);  // 使用cuda核函数获取最大值的索引
    return next;
  }
}
}  // namespace sampler