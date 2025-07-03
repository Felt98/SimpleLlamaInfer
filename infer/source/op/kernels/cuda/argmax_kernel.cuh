#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);     // 获取input_ptr的最大值的索引
}
#endif  // ARGMAX_KERNEL_CUH
