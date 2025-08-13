#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include "model/llama3.h"
#include "benchmark_utils.h"
using namespace std;

struct PerformanceMetrics {
    double first_token_latency_ms;      // 首token延迟(ms)
    double avg_token_latency_ms;        // 平均token延迟(ms)  
    double tokens_per_second;           // 吞吐量(token/s)
    size_t memory_usage_mb;             // 内存使用(MB)
    int total_tokens;                   // 总token数
    double total_time_s;                // 总时间(s)
};

class PerformanceTester {
private:
    const model::LLama3Model& model_;
    
public:
    explicit PerformanceTester(const model::LLama3Model& model) : model_(model) {}
    
    // 基础性能测试
    PerformanceMetrics benchmark_basic(const std::string& prompt, int max_tokens) {
        PerformanceMetrics metrics = {};
        
        auto tokens = model_.encode(prompt);
        int32_t prompt_len = tokens.size();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 首token计时
        auto first_token_start = std::chrono::high_resolution_clock::now();
        int32_t first_token = run_first_token(tokens);
        auto first_token_end = std::chrono::high_resolution_clock::now();
        
        metrics.first_token_latency_ms = std::chrono::duration<double, std::milli>(
            first_token_end - first_token_start).count();
        
        // 后续token生成计时
        std::vector<double> token_latencies;
        std::vector<int32_t> generated_tokens;
        
        for (int i = prompt_len; i < max_tokens; ++i) {
            auto token_start = std::chrono::high_resolution_clock::now();
            int32_t next_token = generate_single_token(first_token, i);
            auto token_end = std::chrono::high_resolution_clock::now();
            
            double latency = std::chrono::duration<double, std::milli>(
                token_end - token_start).count();
            token_latencies.push_back(latency);
            generated_tokens.push_back(next_token);
            
            if (model_.is_sentence_ending(next_token)) {
                break;
            }
            first_token = next_token;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // 计算指标
        metrics.avg_token_latency_ms = std::accumulate(token_latencies.begin(), 
            token_latencies.end(), 0.0) / token_latencies.size();
        metrics.total_time_s = std::chrono::duration<double>(end_time - start_time).count();
        metrics.total_tokens = prompt_len + generated_tokens.size();
        metrics.tokens_per_second = metrics.total_tokens / metrics.total_time_s;
        
        return metrics;
    }
    
    // 设备性能对比测试
    void device_comparison_test(const std::string& prompt, int max_tokens) {
        std::cout << "\n=== 设备性能对比测试 ===\n";
        
        // 注意：这里需要重新初始化模型来切换设备
        // 实际使用时需要创建两个不同的模型实例
        
        std::cout << "CUDA设备测试...\n";
        auto cuda_metrics = benchmark_basic(prompt, max_tokens);
        print_metrics("CUDA", cuda_metrics);
        
        // CPU测试需要重新创建模型实例
        std::cout << "\n注意：CPU测试需要重新初始化模型\n";
    }
    
    // 不同序列长度测试
    void sequence_length_test(const std::string& base_prompt) {
        std::cout << "\n=== 序列长度性能测试 ===\n";
        std::vector<int> seq_lengths = {64, 128, 256, 512, 1024};
        
        for (auto len : seq_lengths) {
            std::cout << "测试序列长度: " << len << " tokens\n";
            auto metrics = benchmark_basic(base_prompt, len);
            print_metrics("Seq_" + std::to_string(len), metrics);
            std::cout << "\n";
        }
    }
    
    // 连续推理压力测试
    void stress_test(const std::string& prompt, int iterations, int max_tokens) {
        std::cout << "\n=== 连续推理压力测试 ===\n";
        std::cout << "执行 " << iterations << " 次推理...\n";
        
        std::vector<PerformanceMetrics> all_metrics;
        auto overall_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            if (i % 10 == 0) {
                std::cout << "Progress: " << i << "/" << iterations << "\n";
            }
            
            auto metrics = benchmark_basic(prompt, max_tokens);
            all_metrics.push_back(metrics);
        }
        
        auto overall_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(overall_end - overall_start).count();
        
        // 统计分析
        print_stress_test_summary(all_metrics, total_time, iterations);
    }

private:
    int32_t run_first_token(const std::vector<int32_t>& tokens) {
        // 实现首token生成逻辑
        // 这里简化实现，实际需要完整的prompt处理流程
        auto prompt_embedding = model_.embedding(tokens);
        tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
        
        int32_t pos = 0;
        int32_t next = -1;
        pos_tensor.index<int32_t>(0) = pos;
        
        // 处理prompt阶段
        for (pos = 0; pos < tokens.size() - 1; ++pos) {
            pos_tensor.index<int32_t>(0) = pos;
            tensor::Tensor input = model_.fill_input(pos_tensor, prompt_embedding, true);
            model_.predict(input, pos_tensor, true, next);
        }
        
        // 生成第一个新token
        std::vector<int32_t> last_token = {tokens.back()};
        auto token_embedding = model_.embedding(last_token);
        tensor::Tensor input = model_.fill_input(pos_tensor, token_embedding, false);
        model_.predict(input, pos_tensor, false, next);
        
        return next;
    }
    
    int32_t generate_single_token(int32_t current_token, int32_t pos) {
        // 生成单个token
        std::vector<int32_t> tokens = {current_token};
        auto token_embedding = model_.embedding(tokens);
        tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
        pos_tensor.index<int32_t>(0) = pos;
        
        tensor::Tensor input = model_.fill_input(pos_tensor, token_embedding, false);
        int32_t next = -1;
        model_.predict(input, pos_tensor, false, next);
        return next;
    }
    
public:
    void print_metrics(const std::string& test_name, const PerformanceMetrics& metrics) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << test_name << "] 性能指标:\n";
        std::cout << "  首Token延迟: " << metrics.first_token_latency_ms << " ms\n";
        std::cout << "  平均Token延迟: " << metrics.avg_token_latency_ms << " ms\n";
        std::cout << "  总Token数: " << metrics.total_tokens << "\n";
        std::cout << "  总耗时: " << metrics.total_time_s << " s\n";
        std::cout << "  吞吐量: " << metrics.tokens_per_second << " tokens/s\n";
    }

private:
    
    void print_stress_test_summary(const std::vector<PerformanceMetrics>& all_metrics,
                                   double total_time, int iterations) {
        // 计算统计指标
        auto calc_stats = [](const std::vector<double>& values) {
            double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double variance = 0.0;
            for (auto val : values) {
                variance += (val - mean) * (val - mean);
            }
            variance /= values.size();
            return std::make_pair(mean, std::sqrt(variance));
        };
        
        std::vector<double> first_token_latencies, avg_token_latencies, throughputs;
        for (const auto& m : all_metrics) {
            first_token_latencies.push_back(m.first_token_latency_ms);
            avg_token_latencies.push_back(m.avg_token_latency_ms);
            throughputs.push_back(m.tokens_per_second);
        }
        
        auto [ft_mean, ft_std] = calc_stats(first_token_latencies);
        auto [at_mean, at_std] = calc_stats(avg_token_latencies);
        auto [tp_mean, tp_std] = calc_stats(throughputs);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n压力测试总结 (" << iterations << " 次推理):\n";
        std::cout << "首Token延迟: " << ft_mean << " ± " << ft_std << " ms\n";
        std::cout << "平均Token延迟: " << at_mean << " ± " << at_std << " ms\n";
        std::cout << "吞吐量: " << tp_mean << " ± " << tp_std << " tokens/s\n";
        std::cout << "总耗时: " << total_time << " s\n";
        std::cout << "平均每次推理: " << total_time / iterations << " s\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        LOG(INFO) << "Usage: ./performance_test checkpoint_path tokenizer_path [--quant]";
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
    
    PerformanceTester tester(model);
    
    std::cout << "=== LLaMA3 推理性能测试 ===\n";
    std::cout << "模型类型: " << (use_quantization ? "量化模型" : "标准模型") << "\n";
    
    const std::string test_prompt = "The future of artificial intelligence is";
    
    // 1. 基础性能测试
    std::cout << "\n1. 基础性能测试\n";
    auto basic_metrics = tester.benchmark_basic(test_prompt, 128);
    tester.print_metrics("Basic", basic_metrics);
    
    // 2. 不同序列长度测试
    tester.sequence_length_test(test_prompt);
    
    // 3. 压力测试
    tester.stress_test(test_prompt, 50, 64);
    
    std::cout << "\n性能测试完成！\n";
    return 0;
} 