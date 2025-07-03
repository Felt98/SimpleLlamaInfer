#ifndef INCLUDE_BASE_ALLOC_H_
#define INCLUDE_BASE_ALLOC_H_
#include <unordered_map>
#include <memory>
#include "base.h"
namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}
  // 获取设备类型
  virtual DeviceType device_type() const { return device_type_; }
  // 释放内存
  virtual void release(void* ptr) const = 0;
  // 分配内存
  virtual void* allocate(size_t byte_size) const = 0;
  // 内存拷贝
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;
  // 内存置零
  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

 private:
  // 设备类型
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();
  // 分配内存
  void* allocate(size_t byte_size) const override;
  // 释放内存
  void release(void* ptr) const override;
};

struct CudaMemoryBuffer {
  void* data;  // 内存指针
  size_t byte_size;  // 内存大小
  bool busy;  // 是否被使用

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  // 未被使用的小内存大小
  mutable std::unordered_map<int, size_t> cuda_no_busy_cnt_;
  // 未被使用的大内存大小
  mutable std::unordered_map<int, size_t> big_no_busy_cnt_;
  // 大内存
  mutable std::unordered_map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  // 小内存
  mutable std::unordered_map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base
#endif  // INCLUDE_BASE_ALLOC_H_