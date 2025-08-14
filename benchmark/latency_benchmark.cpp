#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <iostream>
#include "model/llama3.h"

class LatencyBenchmark {
private:
    const model::LLama3Model& model_;
    
public:
    explicit LatencyBenchmark(const model::LLama3Model& model) : model_(model) {}
    
    // 首Token延迟测试 (TTFT - Time To First Token)
    void test_first_token_latency() {
        std::cout << "\n=== 首Token延迟测试 (TTFT) ===\n";
        
        std::vector<std::string> test_prompts = {
            "Hello",
            "The weather today is",
            "In the field of artificial intelligence, machine learning represents",
            "Climate change is one of the most pressing issues facing humanity today. Scientists worldwide agree that immediate action is required to reduce greenhouse gas emissions and transition to renewable energy sources."
        };
        
        for (size_t i = 0; i < test_prompts.size(); ++i) {
            auto tokens = model_.encode(test_prompts[i]);
            std::cout << "Prompt " << (i+1) << " (长度: " << tokens.size() << " tokens): ";
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // 处理完整prompt + 生成首个新token
            auto prompt_embedding = model_.embedding(tokens);
            tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
            
            // 处理prompt部分
            for (int32_t pos = 0; pos < static_cast<int32_t>(tokens.size()) - 1; ++pos) {
                pos_tensor.index<int32_t>(0) = pos;
                tensor::Tensor input = model_.fill_input(pos_tensor, prompt_embedding, true);
                int32_t next = -1;
                model_.predict(input, pos_tensor, true, next);
            }
            
            // 生成首个新token
            std::vector<int32_t> last_token = {tokens.back()};
            auto token_embedding = model_.embedding(last_token);
            pos_tensor.index<int32_t>(0) = static_cast<int32_t>(tokens.size()) - 1;
            tensor::Tensor input = model_.fill_input(pos_tensor, token_embedding, false);
            int32_t next = -1;
            model_.predict(input, pos_tensor, false, next);
            
            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            
            std::cout << std::fixed << std::setprecision(2) << latency << " ms\n";
        }
    }
    
    // 单Token生成延迟测试
    void test_token_generation_latency() {
        std::cout << "\n=== 单Token生成延迟测试 ===\n";
        
        // 使用固定的prompt进行测试
        std::string prompt = "The future of artificial intelligence";
        auto tokens = model_.encode(prompt);
        auto prompt_embedding = model_.embedding(tokens);
        tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
        
        // 先处理prompt
        for (int32_t pos = 0; pos < static_cast<int32_t>(tokens.size()) - 1; ++pos) {
            pos_tensor.index<int32_t>(0) = pos;
            tensor::Tensor input = model_.fill_input(pos_tensor, prompt_embedding, true);
            int32_t next = -1;
            model_.predict(input, pos_tensor, true, next);
        }
        
        // 生成第一个token
        std::vector<int32_t> current_token = {tokens.back()};
        auto token_embedding = model_.embedding(current_token);
        pos_tensor.index<int32_t>(0) = static_cast<int32_t>(tokens.size()) - 1;
        tensor::Tensor input = model_.fill_input(pos_tensor, token_embedding, false);
        int32_t next = -1;
        model_.predict(input, pos_tensor, false, next);
        
        // 测试后续token生成延迟
        std::vector<double> latencies;
        int32_t current = next;
        
        for (int i = 0; i < 20; ++i) {  // 生成20个token进行测试
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<int32_t> tokens_vec = {current};
            auto embedding = model_.embedding(tokens_vec);
            pos_tensor.index<int32_t>(0) = static_cast<int32_t>(tokens.size()) + i;
            tensor::Tensor input_tensor = model_.fill_input(pos_tensor, embedding, false);
            
            model_.predict(input_tensor, pos_tensor, false, next);
            
            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency);
            
            if (model_.is_sentence_ending(next)) {
                break;
            }
            current = next;
        }
        
        // 统计结果
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_lat = *std::min_element(latencies.begin(), latencies.end());
        double max_lat = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "生成Token数: " << latencies.size() << "\n";
        std::cout << "平均延迟: " << avg << " ms\n";
        std::cout << "最小延迟: " << min_lat << " ms\n";
        std::cout << "最大延迟: " << max_lat << " ms\n";
        std::cout << "标准差: " << calculate_std_dev(latencies, avg) << " ms\n";
    }
    
    // 不同位置的延迟测试
    void test_position_based_latency() {
        std::cout << "\n=== 序列位置延迟分析 ===\n";
        
        std::string prompt = "AI technology has evolved";
        int max_tokens = 100;
        
        auto tokens = model_.encode(prompt);
        auto prompt_embedding = model_.embedding(tokens);
        tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
        
        // 处理prompt
        for (int32_t pos = 0; pos < static_cast<int32_t>(tokens.size()) - 1; ++pos) {
            pos_tensor.index<int32_t>(0) = pos;
            tensor::Tensor input = model_.fill_input(pos_tensor, prompt_embedding, true);
            int32_t next = -1;
            model_.predict(input, pos_tensor, true, next);
        }
        
        std::vector<double> position_latencies;
        int32_t current = tokens.back();
        
        for (int pos = static_cast<int>(tokens.size()); pos < max_tokens; ++pos) {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<int32_t> tokens_vec = {current};
            auto embedding = model_.embedding(tokens_vec);
            pos_tensor.index<int32_t>(0) = pos - 1;
            tensor::Tensor input_tensor = model_.fill_input(pos_tensor, embedding, false);
            
            int32_t next = -1;
            model_.predict(input_tensor, pos_tensor, false, next);
            
            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            position_latencies.push_back(latency);
            
            if (model_.is_sentence_ending(next)) {
                break;
            }
            current = next;
            
            // 每10个位置打印一次
            if ((pos - static_cast<int>(tokens.size())) % 10 == 0) {
                std::cout << "Position " << pos << ": " << std::fixed << std::setprecision(2) 
                          << latency << " ms\n";
            }
        }
        
        // 分析延迟趋势
        analyze_latency_trend(position_latencies);
    }

private:
    double calculate_std_dev(const std::vector<double>& values, double mean) {
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        return std::sqrt(variance);
    }
    
    void analyze_latency_trend(const std::vector<double>& latencies) {
        std::cout << "\n延迟趋势分析:\n";
        
        if (latencies.size() < 20) {
            std::cout << "样本不足，无法分析趋势\n";
            return;
        }
        
        // 计算前半部分和后半部分的平均延迟
        size_t mid = latencies.size() / 2;
        double first_half = std::accumulate(latencies.begin(), latencies.begin() + mid, 0.0) / mid;
        double second_half = std::accumulate(latencies.begin() + mid, latencies.end(), 0.0) / (latencies.size() - mid);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "前半部分平均延迟: " << first_half << " ms\n";
        std::cout << "后半部分平均延迟: " << second_half << " ms\n";
        
        double trend = ((second_half - first_half) / first_half) * 100;
        std::cout << "延迟变化趋势: " << (trend > 0 ? "+" : "") << trend << "%\n";
        
        if (trend > 5) {
            std::cout << "⚠️  延迟随序列长度显著增长\n";
        } else if (trend < -5) {
            std::cout << "⬇️  延迟随序列长度显著下降\n";
        } else {
            std::cout << "✅ 延迟相对稳定\n";
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        LOG(INFO) << "Usage: ./latency_benchmark checkpoint_path tokenizer_path [--quant]";
        LOG(INFO) << "  --quant: 使用量化模型 (可选，默认不使用量化)";
        return -1;
    }
    
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    
    // 检查是否使用量化模型
    bool use_quantization = false;
    if (argc == 4 && std::string(argv[3]) == "--quant") {
        use_quantization = true;
        std::cout << "使用量化模型模式\n";
    } else {
        std::cout << "使用标准模型模式\n";
    }
    
    // 初始化模型
    model::LLama3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                             checkpoint_path, use_quantization);
    
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "模型初始化失败: " << init_status.get_err_code();
        return -1;
    }
    
    LatencyBenchmark benchmark(model);
    
    std::cout << "=== LLaMA3 延迟性能详细测试 ===\n";
    std::cout << "模型类型: " << (use_quantization ? "量化模型" : "标准模型") << "\n";
    
    // 1. 首Token延迟测试
    benchmark.test_first_token_latency();
    
    // 2. Token生成延迟测试
    benchmark.test_token_generation_latency();
    
    // 3. 位置相关延迟测试
    benchmark.test_position_based_latency();
    
    std::cout << "\n延迟测试完成！\n";
    return 0;
} 