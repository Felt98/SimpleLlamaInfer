#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
int32_t generate(const model::LLama3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  // 编码句子，tokens是句子编码后的id列表vector<int32_t>
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();  // 句子长度
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;  // 当前位置
  int32_t next = -1;  // 下一个token
  bool is_prompt = true;  // 是否是prompt

  // 获取prompt embedding
  const auto& prompt_embedding = model.embedding(tokens);

  // 获取pos张量，用于存储当前位置，维度为(1)
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  // words数组：模型输出结果
  std::vector<int32_t> words;

  // 循环生成
  while (pos < total_steps) {
    // 设置pos_tensor为pos
    pos_tensor.index<int32_t>(0) = pos;

    // 处理prompt：如果pos小于prompt长度-1，则处理prompt
    if (pos < prompt_len - 1) {
      // input指向第pos个token的embedding
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      // 这里的预测主要是用于计算prompt的kv cache，因此这里的next没有用（next为-1）
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      // 处理完prompt后，开始处理模型的输出
      // 设置is_prompt为false
      is_prompt = false;

      // next作为下一个输入token
      tokens = std::vector<int32_t>{next};
      // 获取当前输入token的embedding
      const auto& token_embedding = model.embedding(tokens);
      // input指向第0个token的embedding （pos_tensor没有用了）
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      // 预测下一个token
      model.predict(input, pos_tensor, is_prompt, next);
    }
    // 如果句子结束，则退出循环
    if (model.is_sentence_ending(next)) {
      break;
    }
    
    if (is_prompt) {
      // 如果is_prompt为true，则next为下一个输入token
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      // 如果is_prompt为false，则next不为下一个输入token
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

  // model::LLama3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
  //   checkpoint_path, false);  // true为使用量化模型，false为使用fp16模型,llama3.1使用bpe分词
  model::LLama3Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
    checkpoint_path, false);  // true为使用量化模型，false为使用fp16模型，llama2使用spe分词

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
