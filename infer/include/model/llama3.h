#ifndef INCLUDE_MODEL_LLAMA_H_
#define INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLama2Layers {
  // 1. 基本层（不带权重）：残差连接、旋转位置编码、SwiGLU激活函数、多头注意力层
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  // 2. 多头注意力层（带权重）：Wq、Wk、Wv、Wo
  std::vector<std::shared_ptr<op::Layer>> wq_layers_; // 查询权重矩阵
  std::vector<std::shared_ptr<op::Layer>> wk_layers_; // 键权重矩阵
  std::vector<std::shared_ptr<op::Layer>> wv_layers_; // 值权重矩阵
  std::vector<std::shared_ptr<op::Layer>> wo_layers_; // 输出权重矩阵

  // 3. 前馈神经网络层（带权重）：W1、W2、RMSNorm、W3
  std::vector<std::shared_ptr<op::Layer>> w1_layers_; // 前馈神经网络层1，一个层可能有多个权重
  std::vector<std::shared_ptr<op::Layer>> w2_layers_; // 前馈神经网络层2
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_; // rmsnorm层
  std::vector<std::shared_ptr<op::Layer>> w3_layers_; // 前馈神经网络层3

  // 4. 分类层（带权重）：全连接层
  std::shared_ptr<op::Layer> cls_layer_;

  // 5. 嵌入层（带权重）：词嵌入层
  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLama2Model : public Model {
 public:
  // 1. 构造函数：初始化模型类型、词表路径、模型路径、是否量化
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  // 2. 初始化：设置设备类型、初始化内存、创建层、创建参数层、创建非参数层、创建量化层
  base::Status init(base::DeviceType device_type) override;

  // 3. 预测：预测下一个token
  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  // 4. 前向传播：前向传播
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  // 5. 嵌入：嵌入
  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  // 6. 初始化内存：初始化内存
  void init_mem() override;

  // 7. 创建层：创建层
  base::Status create_layers() override;

  // 8. 创建参数层：创建参数层
  void create_param_layers() override;

  // 9. 创建非参数层：创建非参数层
  void create_nonparam_layers() override;

  // 10. 创建量化层：创建量化层
  void create_param_quant_layers() override;

  // 11. 注意力：mha注意力
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  // 12. 注意力：rms注意力
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  // 13. 前馈神经网络：前馈神经网络
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  // 14. 注意力：qkv注意力
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  // 15. 分类：分类
  void cls_logits(const tensor::Tensor& input) const;

  // 16. 后处理：后处理
  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  // cuda配置
  std::unique_ptr<LLama2Layers> llama_layers_;  // llama层  
};
}  // namespace model

#endif