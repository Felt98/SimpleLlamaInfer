#include "../cpu/mha_kernel.h"
#include <cuda_runtime_api.h>
#include "../kernels_interface.h"
namespace kernel {
  /*
  mha_kernel 函数是用于实现GQA多头注意力机制的核函数。
  参数说明：
  pos: 当前位置
  head_num: 头数
  layer_index: 层索引
  seq_len: 序列长度
  kv_dim: 键值维度
  kv_mul: GQA的组数
  head_size: 头维度
  mha_out: 输出
  query_tensor: 查询
  score_tensor: 得分
  key_cache_tensor: 键缓存
  value_cache_tensor: 值缓存
  device_type: 设备类型
  config: 配置
  */
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type == base::DeviceType::kDeviceCPU) {
      allocator = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
      allocator = base::CUDADeviceAllocatorFactory::get_instance();
    }
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len); // 当前head 对应的score_tensor
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    
    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                               query_head_addr);
    query_mat.set_device_type(device_type);
    
    for (int32_t t = 0; t <= pos; t++) {
      int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
      // token t对应的key_cache_tensor
      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));
      
      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      // 计算query_mat和key_mat的乘积，得到score_mat
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    int32_t cache_offset = (h / kv_mul) * head_size;
    float* value_head_addr =
        const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
    tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                value_head_addr);
    get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                      head_size, kv_dim, config ? config->stream : nullptr);
  }
}
}  // namespace kernel