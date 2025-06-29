#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  // 编码句子
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  // 获取prompt embedding
  const auto& prompt_embedding = model.embedding(tokens);
  // 获取pos张量
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  // 创建words数组
  std::vector<int32_t> words;
  // 循环生成
  while (pos < total_steps) {
    // 设置pos张量
    pos_tensor.index<int32_t>(0) = pos;
    // 如果pos小于prompt长度-1，则填充输入
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      // 预测
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      // 设置is_prompt为false
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      // 获取token embedding
      const auto& token_embedding = model.embedding(tokens);
      // 填充输入
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      // 预测
      model.predict(input, pos_tensor, is_prompt, next);
    }
    // 如果句子结束，则退出循环
    if (model.is_sentence_ending(next)) {
      break;
    }
    // 如果is_prompt为true，则添加下一个token
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      // 如果is_prompt为false，则添加下一个token
      words.push_back(next);
    }

    pos += 1;
  }
  // 如果需要输出，则输出结果
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  // 返回结果
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  model::LLama2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
    checkpoint_path, false);  // true为使用量化模型，false为使用fp16模型
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  const std::string& sentence = "hello";

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
