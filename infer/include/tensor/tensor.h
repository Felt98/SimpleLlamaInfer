#ifndef INCLUDE_TENSOR_TENSOR_H_
#define INCLUDE_TENSOR_TENSOR_H_
#include <driver_types.h>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {

class Tensor {
 public:
  explicit Tensor() = default;

  // 一维
  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
  // 二维
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
  // 三维
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  // 四维   
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 多维
  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
  // 将张量复制到cpu
  void to_cpu();
  // 将张量复制到cuda
  void to_cuda(cudaStream_t stream = nullptr);
  // 判断张量是否为空
  bool is_empty() const;
  // 初始化buffer
  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                   bool need_alloc, void* ptr);
  // 获取张量指针
  template <typename T>
  T* ptr();
  // 获取const张量指针
  template <typename T>
  const T* ptr() const;
  // 重塑张量
  void reshape(const std::vector<int32_t>& dims);
  // 获取buffer_数据指针
  std::shared_ptr<base::Buffer> get_buffer() const;
  // 获取张量大小
  size_t size() const;
  // 获取张量字节大小
  size_t byte_size() const;
  // 获取张量维度大小
  int32_t dims_size() const;
  // 获取张量数据类型
  base::DataType data_type() const;
  // 获取张量维度
  int32_t get_dim(int32_t idx) const;
  // 获取张量维度
  const std::vector<int32_t>& dims() const;
  // 获取张量步长
  std::vector<size_t> strides() const;
  // 赋值buffer到Tensor的buffer_
  bool assign(std::shared_ptr<base::Buffer> buffer);

  // 重置张量
  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  // 设置设备类型
  void set_device_type(base::DeviceType device_type) const;
  // 获取设备类型
  base::DeviceType device_type() const;

  // 分配内存
  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  // 获取起始位置为index的张量指针
  template <typename T>
  T* ptr(int64_t index);
  // 获取起始位置为index的const张量指针
  template <typename T>
  const T* ptr(int64_t index) const;

  // 获取buffer_[offset]的值
  template <typename T>
  T& index(int64_t offset);

  // 获取buffer_[offset]的值
  template <typename T>
  const T& index(int64_t offset) const;

  // 克隆张量
  tensor::Tensor clone() const;

 private:
  size_t size_ = 0;   // 张量数据总大小，即所有元素个数
  std::vector<int32_t> dims_;  // 张量维度
  std::shared_ptr<base::Buffer> buffer_;  // 内存资源，使用shared_ptr管理，如果想让buffer_独立，应该用 clone() 或自定义深拷贝。
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;  // 数据类型
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  // 返回当前buffer的指针，并将其void*转换为T*类型
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;  
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}
}  // namespace tensor
#endif  // INCLUDE_TENSOR_TENSOR_H_
