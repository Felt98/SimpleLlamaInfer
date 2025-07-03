#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

// 检查当前系统是否支持 POSIX.1-2001 标准
#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }

#ifdef HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  // 根据内存大小选择对齐方式，如果内存大小大于 1024 字节，则对齐方式为 32 字节，否则为 16 字节
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  // 使用 posix_memalign 分配内存，并指定对齐方式 32 或 16 字节
  int status = posix_memalign((void**)&data,
                              (std::max(alignment, sizeof(void*))),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  // 使用 malloc 分配内存
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base