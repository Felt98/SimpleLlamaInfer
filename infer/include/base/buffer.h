#ifndef INCLUDE_BASE_BUFFER_H_
#define INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();
  // 分配内存
  bool allocate();

  // 从另一个Buffer拷贝数据
  void copy_from(const Buffer& buffer) const;

  // 从另一个Buffer拷贝数据
  void copy_from(const Buffer* buffer) const;

  // 获取内存指针
  void* ptr();

  // 获取内存指针
  const void* ptr() const;

  // 获取内存大小
  size_t byte_size() const;

  // 获取分配器
  std::shared_ptr<DeviceAllocator> allocator() const;

  // 获取设备类型
  DeviceType device_type() const;

  // 设置设备类型
  void set_device_type(DeviceType device_type);

  // 获取返回一个指向当前对象的共享指针
  // 调用enable_shared_from_this的shared_from_this()方法
  std::shared_ptr<Buffer> get_shared_from_this();

  // 是否是外部内存
  bool is_external() const;
};
}  // namespace base

#endif