#ifndef INCLUDE_MODEL_MODEL_H_
#define INCLUDE_MODEL_MODEL_H_
#include <op/embedding.h>
#include <map>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {
class Model {
 public:
  // 1. 构造函数：初始化模型类型、词表路径、模型路径、是否量化
  // tokenizer_type: 词表类型 （SPE、BPE）
  // model_type: 模型类型 （LLAMA2、LLAMA3、QWEN2.5、QWEN3）
  // token_path: 词表路径（构建encoder时，需要词表路径）
  // model_path: 模型路径
  // is_quant_model: 是否量化
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  // 2. 初始化：设置设备类型、初始化内存、创建层、创建参数层、创建非参数层、创建量化层
  virtual base::Status init(base::DeviceType device_type) = 0;

  // 3. 预测：预测下一个token
  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  // 4. 前向传播：前向传播
  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  // 5. 获取模型类型
  base::ModelType model_type() const;

  // 6. 获取词表路径
  const std::string& token_path() const;

  // 7. 获取模型路径
  const std::string& model_path() const;

  // 8. 从buffers_获取第buffer_idx个tensor
  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  // 9. 从buffers_获取第buffer_idx个tensor
  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  // 10. 判断是否结束
  virtual bool is_sentence_ending(int32_t token_idx) const;

  // 11. 解码：解码单个token
  virtual std::string decode(int32_t token_idx) const;

  // 12. 解码：解码多个token
  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  // 13. 编码：编码句子
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  // 14. 切片KV缓存：获取第token_pos步时，第layer_idx层的KV-Cache张量
  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  // 15. 嵌入：嵌入
  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  // 16. 填充输入：填充输入
  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

 protected:
  // 17. 插入缓冲区：插入缓冲区
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  // 17. 读取模型文件：读取模型文件
  virtual base::Status read_model_file();

  // 18. 创建编码层：创建编码层
  virtual base::Status create_encode_layer();

  // 19. 生成模型：生成模型
  virtual base::Status gen_model_from_file();

  // 20. 生成模型信息：生成模型信息
  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  // 21. 后处理：后处理
  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

  // 22. 初始化内存：初始化内存
 private:
  virtual void init_mem() = 0;

  // 23. 创建层：创建层
  virtual base::Status create_layers() = 0;

  // 24. 创建参数层：创建参数层
  virtual void create_param_layers() = 0;
  // 25. 创建非参数层：创建非参数层
  virtual void create_nonparam_layers() = 0;

  // 26. 创建量化层：创建量化层
  virtual void create_param_quant_layers() = 0;

 protected:
  
  int32_t group_size_ = 1;   // 组大小
  bool is_quant_model_ = false;  // 是否量化
  std::unique_ptr<TransformerConfig> config_;  // 配置

  std::string token_path_;  // 词表路径（构建encoder时，需要词表路径）
  std::string model_path_;  // 模型路径
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;  // 编码层
  std::map<ModelBufferType, tensor::Tensor> buffers_;  // Tensor缓冲区哈希表，存取buffer_idx和tensor
  std::unique_ptr<sampler::Sampler> sampler_;  // 采样器
  std::shared_ptr<RawModelData> raw_model_data_;  // 原始模型数据
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  // 设备类型
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;  // 模型类型
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;  // 词表类型
};
}  // namespace model
#endif  // INCLUDE_MODEL_MODEL_H_
