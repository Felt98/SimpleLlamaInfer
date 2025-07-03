#ifndef INCLUDE_OP_ENCODE_H_
#define INCLUDE_OP_ENCODE_H_
#include <sentencepiece_processor.h>
#include "layer.h"
#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#endif
namespace op {

class EncodeLayerBase : public Layer {
 public:
  explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
      : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
        has_bos_(has_bos),
        has_eos_(has_eos),
        token_model_path_(std::move(token_model_path)) {}

  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  virtual std::string decode(int32_t token_id) const = 0;

  virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

  virtual bool is_sentence_ending(int32_t token_id) const = 0;

  virtual int32_t vocab_size() const = 0;

 protected:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::string token_model_path_;
};

// 使用sentencepiece库的EncodeLayer （llama2支持SPE）
class SpeEncodeLayer : public EncodeLayerBase {
 public:
  explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

  // 编码函数，使用sentencepiece库将句子编码为id列表
  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::string decode(int32_t token_id) const override;

  std::string decode(const std::vector<int32_t>& token_ids) const override;

  bool is_sentence_ending(int32_t token_id) const override;

  // 获取词表大小
  int32_t vocab_size() const override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe; // 句柄，使用sentencepiece库
};

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
// 使用tiktoken库的EncodeLayer （llama3、qwen2.5、qwen3支持BPE）
class BpeEncodeLayer : public EncodeLayerBase {
public:
  explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::string decode(int32_t token_id) const override;

  std::string decode(const std::vector<int32_t>& token_ids) const override;

  bool is_sentence_ending(int32_t token_id) const override;

  int32_t vocab_size() const override;

 protected:
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t stop_token1_ = -1;
  int32_t stop_token2_ = -1;
  int32_t num_token_ = 0;
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenEncodeLayer : public BpeEncodeLayer {
public:
  explicit QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
};
#endif

}  // namespace op
#endif  // INCLUDE_OP_ENCODE_H_
