#ifndef INCLUDE_OP_LAYER_H_
#define INCLUDE_OP_LAYER_H_
#include <cstdint>
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

// 无参数的层
class Layer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "",base::DataType data_type = base::DataType::kDataTypeFp32);

  base::DataType data_type() const;
  LayerType layer_type() const;
  virtual ~Layer() = default;
  virtual base::Status init();
  virtual base::Status forward();
  virtual base::Status forward(const std::vector<tensor::Tensor>& inputs, const std::vector<tensor::Tensor>& outputs);
  
  // 以下是重载的forward方法，用于兼容旧的接口
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1);
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output1);
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& output1);
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& output1);
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& input5, const tensor::Tensor& output1);

  virtual void set_input(int32_t idx, const tensor::Tensor& input);
  virtual void set_output(int32_t idx, const tensor::Tensor& output);
  virtual size_t input_size() const;
  virtual size_t output_size() const;
  virtual base::Status check() const;
  virtual tensor::Tensor& get_input(int32_t idx);
  virtual tensor::Tensor& get_output(int32_t idx);
  virtual const tensor::Tensor& get_input(int32_t idx) const ;
  virtual const tensor::Tensor& get_output(int32_t idx) const ;
  virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);
  virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

  const std::string& get_layer_name() const;
  void set_layer_name(const std::string& layer_name);
  base::DeviceType device_type() const;
  void set_device_type(base::DeviceType device_type);

  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const;
  base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type, ...) const;

  void reset_input_size(size_t size);
  void reset_output_size(size_t size);
  virtual void to_cuda();
  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);
  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

 protected:
  std::string layer_name_;  // 层名称
  LayerType layer_type_ = LayerType::kLayerUnknown;  // 层类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;  // 数据类型
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  // 设备类型
  std::vector<tensor::Tensor> inputs_;  // 输入张量
  std::vector<tensor::Tensor> outputs_;  // 输出张量
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  // cuda配置
};

// 有参数的层
class LayerParam : public Layer {
 public:
  explicit LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer = false, std::string layer_name = "",base::DataType data_type = base::DataType::kDataTypeFp32);
  virtual ~LayerParam() = default;

  size_t weight_size() const;
  void reset_weight_size(size_t size);
  tensor::Tensor& get_weight(int32_t idx);
  const tensor::Tensor& get_weight(int32_t idx) const;
  void to_cuda() override;
  base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;
  base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;
  void set_scales(const tensor::Tensor& scales);
  void set_group_size(int32_t group_size);
  int32_t get_scale_num() const;
 protected:
  int32_t group_size_ = 0;  // 量化组大小
  bool is_quant_layer_ = false;  // 是否量化层
  tensor::Tensor scales_;  // 缩放因子张量
  std::vector<tensor::Tensor> weights_;  // 权重张量数组，一个层可能有多个权重
};
}  // namespace op
#endif  // INCLUDE_OP_LAYER_H_
