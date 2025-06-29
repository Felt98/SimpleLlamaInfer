#include "model/raw_model_data.h"
#include <sys/mman.h>
#include <unistd.h>
namespace model {
RawModelData::~RawModelData() {
  // 如果数据不为空且不是MAP_FAILED，则释放内存 
  if (data != nullptr && data != MAP_FAILED) {
    // munmap释放mmap内存
    munmap(data, file_size);
    data = nullptr;
  }
  // 如果文件描述符不为-1，则关闭文件描述符
  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

// 获取fp32权重
const void* RawModelDataFp32::weight(size_t offset) const {
  return static_cast<float*>(weight_data) + offset;
}

// 获取int8权重
const void* RawModelDataInt8::weight(size_t offset) const {
  return static_cast<int8_t*>(weight_data) + offset;
}
}  // namespace model