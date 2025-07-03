#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  // 如果内存大小大于 1MB，则分配大内存
  if (byte_size > 1024 * 1024) {
    std::vector<CudaMemoryBuffer>& big_buffers = this->big_buffers_map_[id];
    int sel_id = -1;
    // 遍历大内存池，找到合适的内存
    for (int i = 0; i < big_buffers.size(); i++) {
      auto& big_buffer=big_buffers[i];
      // 如果内存大小大于等于要分配的内存大小，并且内存未被使用，并且内存大小与要分配的内存大小相差小于 1MB，则选择该内存
      if (big_buffer.byte_size >= byte_size && !big_buffer.busy &&
          big_buffer.byte_size - byte_size < 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffer.byte_size) {
          sel_id = i;
        }
      }
    }

    // 如果找到合适的大内存，则返回
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;      // 标记内存已使用
      big_no_busy_cnt_[id] -= big_buffers[sel_id].byte_size; // 更新未被使用的大内存大小
      return big_buffers[sel_id].data;      // 返回内存指针
    }
    // 如果没找到合适的大内存，则分配新内存
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);

    // 如果分配失败，则返回空指针
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    // 如果分配成功，则将内存添加到大内存池中
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  // 如果内存大小小于 1MB ，则分配小内存
  std::vector<CudaMemoryBuffer>& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    auto& cuda_buffer = cuda_buffers[i];
    if (cuda_buffer.byte_size >= byte_size && !cuda_buffer.busy) {
      cuda_buffer.busy = true;
      cuda_no_busy_cnt_[id] -= cuda_buffer.byte_size;  // 更新未被使用的内存大小
      return cuda_buffer.data;
    }
  }
  // 如果没找到合适的内存，则分配新内存
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  // 如果分配成功，则将内存添加到内存池中
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

// 释放空闲的内存
void release_idle_buffers(std::vector<CudaMemoryBuffer>& buffers){
      std::vector<CudaMemoryBuffer> temp;
      for (auto& buffer: buffers) {
        if (!buffer.busy) {  // 如果内存未被使用，则释放内存
          auto state = cudaFree(buffer.data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device ";
        } else {
          // 如果内存被使用，则将内存添加到临时内存池中
          temp.push_back(buffer);
        }
      }
      buffers=std::move(temp);  // 将临时内存池赋值给小内存池
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    return;
  }
  cudaError_t state = cudaSuccess;

  for (auto& [device_id,cuda_buffers] : cuda_buffers_map_) {
    // 标记要释放的小内存为空闲
    for (auto& cuda_buffer: cuda_buffers) {
      if (cuda_buffer.data == ptr) {  // 如果内存指针相等，则标记该内存为空闲
        cuda_no_busy_cnt_[device_id] += cuda_buffer.byte_size;  // 增加未被使用的内存大小
        cuda_buffer.busy = false;
        return;
      }
    }
  }

  for(auto& [_,big_buffers]:big_buffers_map_){
    // 标记要释放的大内存为空闲
    for (auto& big_buffer : big_buffers) {
      if (big_buffer.data == ptr) {  // 如果内存指针相等，则标记该内存为空闲
        big_buffer.busy = false;
        big_no_busy_cnt_[_] += big_buffer.byte_size; // 增加未被使用的大内存大小
        return;
      }
    }
  }
  // ptr不在内存池中，则直接释放内存
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";


  // 释放小内存，如果未被使用的小内存块大小总和大于 1GB，则释放小内存
  for (auto& [device_id,cuda_buffers] : cuda_buffers_map_) {
    if (cuda_no_busy_cnt_[device_id] > 1024 * 1024 * 1024) {
      release_idle_buffers(cuda_buffers); // 释放空闲的小内存
      cuda_no_busy_cnt_[device_id] = 0;
    }
  }
  // 释放大内存，如果未被使用的大内存块大小总和大于 1GB，则释放大内存
  for (auto& [device_id, big_buffers] : big_buffers_map_) {
    if (big_no_busy_cnt_[device_id] > 1024 * 1024 * 1024) {
      release_idle_buffers(big_buffers); // 释放空闲的大内存
      big_no_busy_cnt_[device_id] = 0;
    }
  }
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base