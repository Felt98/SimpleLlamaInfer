#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include "model/llama3.h"

// 添加CUDA头文件用于GPU内存监控
#include <cuda_runtime.h>
#include <cuda.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/resource.h>
#endif

class MemoryBenchmark {
private:
    const model::LLama3Model& model_;
    size_t baseline_cpu_memory_mb_;
    size_t baseline_gpu_memory_mb_;
    size_t model_gpu_memory_mb_;  // 模型本身占用的GPU内存
    
public:
    explicit MemoryBenchmark(const model::LLama3Model& model) : model_(model) {
        // 记录模型加载后的基准内存使用
        baseline_cpu_memory_mb_ = get_memory_usage_mb();
        baseline_gpu_memory_mb_ = get_gpu_memory_usage_mb();
        
        // 注意：这里无法准确获取模型本身的GPU内存占用
        // 因为模型在构造函数中已经加载完成
        // 建议在main函数中模型初始化前后分别测量
        model_gpu_memory_mb_ = 0; // 需要外部传入
    }
    
    // 设置模型占用的GPU内存（在main函数中调用）
    void set_model_gpu_memory(size_t model_memory_mb) {
        model_gpu_memory_mb_ = model_memory_mb;
    }
    
    // 获取当前进程内存使用 (Linux)
    size_t get_memory_usage_mb() {
#ifdef __linux__
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string label, value, unit;
                iss >> label >> value >> unit;
                return std::stoll(value) / 1024; // 转换为MB
            }
        }
#endif
        return 0;
    }
    
    // 获取GPU内存使用情况
    size_t get_gpu_memory_usage_mb() {
        size_t free_memory, total_memory;
        cudaError_t result = cudaMemGetInfo(&free_memory, &total_memory);
        if (result != cudaSuccess) {
            std::cerr << "CUDA内存信息获取失败: " << cudaGetErrorString(result) << std::endl;
            return 0;
        }
        return (total_memory - free_memory) / (1024 * 1024); // 转换为MB
    }
    
    // 获取GPU总内存
    size_t get_gpu_total_memory_mb() {
        size_t free_memory, total_memory;
        cudaError_t result = cudaMemGetInfo(&free_memory, &total_memory);
        if (result != cudaSuccess) {
            std::cerr << "CUDA内存信息获取失败: " << cudaGetErrorString(result) << std::endl;
            return 0;
        }
        return total_memory / (1024 * 1024); // 转换为MB
    }
    
    // 获取相对于基准的内存变化
    size_t get_cpu_memory_change_mb() {
        return get_memory_usage_mb() - baseline_cpu_memory_mb_;
    }
    
    size_t get_gpu_memory_change_mb() {
        return get_gpu_memory_usage_mb() - baseline_gpu_memory_mb_;
    }
    
    // 基础内存使用测试
    void test_base_memory_usage() {
        std::cout << "\n=== 基础内存使用测试 ===\n";
        
        size_t current_cpu_memory = get_memory_usage_mb();
        size_t current_gpu_memory = get_gpu_memory_usage_mb();
        size_t gpu_total = get_gpu_total_memory_mb();
        
        std::cout << "模型加载后内存使用:\n";
        std::cout << "  CPU内存: " << current_cpu_memory << " MB (基准: " << baseline_cpu_memory_mb_ << " MB)\n";
        std::cout << "  GPU总内存: " << current_gpu_memory << " MB / " << gpu_total << " MB (" 
                  << std::fixed << std::setprecision(1) 
                  << (static_cast<double>(current_gpu_memory) / gpu_total * 100) << "%)\n";
        std::cout << "  GPU基准内存: " << baseline_gpu_memory_mb_ << " MB\n";
        
        if (model_gpu_memory_mb_ > 0) {
            std::cout << "  🎯 模型本身占用GPU内存: " << model_gpu_memory_mb_ << " MB\n";
            std::cout << "  📊 其他进程占用GPU内存: " << (current_gpu_memory - model_gpu_memory_mb_) << " MB\n";
        } else {
            std::cout << "  ⚠️  无法准确计算模型GPU内存占用（可能GPU上有其他进程）\n";
        }
        
        // 注意：由于config_是私有成员，这里使用估算值
        // 基于LLaMA3的典型配置进行估算
        const int estimated_layer_num = 32;
        const int estimated_seq_len = 2048;
        const int estimated_kv_dim = 1024;  // 典型GQA配置
        
        size_t kv_cache_size_mb = (estimated_layer_num * estimated_seq_len * estimated_kv_dim * sizeof(float) * 2) / (1024 * 1024);
        std::cout << "KV Cache估计大小: " << kv_cache_size_mb << " MB (基于典型配置)\n";
    }
    
    // 序列长度对内存影响测试
    void test_sequence_length_memory() {
        std::cout << "\n=== 序列长度内存影响测试 ===\n";
        
        std::vector<int> seq_lengths = {64, 128, 256, 512, 1024};
        std::string base_prompt = "Memory usage test prompt";
        
        for (int seq_len : seq_lengths) {
            size_t cpu_memory_before = get_memory_usage_mb();
            size_t gpu_memory_before = get_gpu_memory_usage_mb();
            
            // 生成指定长度的序列
            generate_sequence(base_prompt, seq_len);
            
            size_t cpu_memory_after = get_memory_usage_mb();
            size_t gpu_memory_after = get_gpu_memory_usage_mb();
            size_t cpu_memory_diff = cpu_memory_after - cpu_memory_before;
            size_t gpu_memory_diff = gpu_memory_after - gpu_memory_before;
            
            std::cout << "序列长度 " << seq_len << ":\n";
            std::cout << "  CPU内存变化: " << cpu_memory_diff << " MB (相对基准: " << get_cpu_memory_change_mb() << " MB)\n";
            std::cout << "  GPU内存变化: " << gpu_memory_diff << " MB (相对基准: " << get_gpu_memory_change_mb() << " MB)\n";
        }
    }
    
    // KV Cache增长测试
    void test_kv_cache_growth() {
        std::cout << "\n=== KV Cache增长测试 ===\n";
        
        std::string prompt = "Testing KV cache memory growth";
        auto tokens = model_.encode(prompt);
        
        std::vector<size_t> cpu_memory_changes;
        std::vector<size_t> gpu_memory_changes;
        std::vector<int> positions;
        
        // 处理prompt并生成tokens，定期记录内存使用
        auto prompt_embedding = model_.embedding(tokens);
        tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
        
        cpu_memory_changes.push_back(get_cpu_memory_change_mb());
        gpu_memory_changes.push_back(get_gpu_memory_change_mb());
        positions.push_back(0);
        
        // 处理prompt阶段
        for (int32_t pos = 0; pos < static_cast<int32_t>(tokens.size()) - 1; ++pos) {
            pos_tensor.index<int32_t>(0) = pos;
            tensor::Tensor input = model_.fill_input(pos_tensor, prompt_embedding, true);
            int32_t next = -1;
            model_.predict(input, pos_tensor, true, next);
            
            if (pos % 10 == 0) {
                cpu_memory_changes.push_back(get_cpu_memory_change_mb());
                gpu_memory_changes.push_back(get_gpu_memory_change_mb());
                positions.push_back(pos);
            }
        }
        
        // 生成阶段 
        int32_t current = tokens.back();
        for (int i = 0; i < 100; ++i) {
            std::vector<int32_t> tokens_vec = {current};
            auto embedding = model_.embedding(tokens_vec);
            pos_tensor.index<int32_t>(0) = static_cast<int32_t>(tokens.size()) + i;
            tensor::Tensor input_tensor = model_.fill_input(pos_tensor, embedding, false);
            
            int32_t next = -1;
            model_.predict(input_tensor, pos_tensor, false, next);
            
            if (i % 10 == 0) {
                cpu_memory_changes.push_back(get_cpu_memory_change_mb());
                gpu_memory_changes.push_back(get_gpu_memory_change_mb());
                positions.push_back(static_cast<int>(tokens.size()) + i);
            }
            
            if (model_.is_sentence_ending(next)) {
                break;
            }
            current = next;
        }
        
        // 分析内存增长趋势
        std::cout << "内存变化趋势 (相对于模型加载后):\n";
        for (size_t i = 0; i < cpu_memory_changes.size(); ++i) {
            std::cout << "Position " << positions[i] << ":\n";
            std::cout << "  CPU变化: " << cpu_memory_changes[i] << " MB";
            if (i > 0) {
                int cpu_diff = static_cast<int>(cpu_memory_changes[i]) - static_cast<int>(cpu_memory_changes[i-1]);
                std::cout << " (本次: " << cpu_diff << ")";
            }
            std::cout << "\n";
            
            std::cout << "  GPU变化: " << gpu_memory_changes[i] << " MB";
            if (i > 0) {
                int gpu_diff = static_cast<int>(gpu_memory_changes[i]) - static_cast<int>(gpu_memory_changes[i-1]);
                std::cout << " (本次: " << gpu_diff << ")";
            }
            std::cout << "\n";
        }
    }
    
    // 内存泄漏检测
    void test_memory_leak() {
        std::cout << "\n=== 内存泄漏检测 ===\n";
        
        std::cout << "初始状态 (模型加载后):\n";
        std::cout << "  CPU基准: " << baseline_cpu_memory_mb_ << " MB\n";
        std::cout << "  GPU基准: " << baseline_gpu_memory_mb_ << " MB\n";
        
        // 执行多次推理循环
        std::string test_prompt = "Memory leak detection test";
        for (int iteration = 0; iteration < 10; ++iteration) {
            for (int i = 0; i < 10; ++i) {
                auto tokens = model_.encode(test_prompt);
                auto embedding = model_.embedding(tokens);
                
                // 简单的推理过程
                tensor::Tensor pos_tensor = model_.get_buffer(model::ModelBufferType::kInputPos);
                pos_tensor.index<int32_t>(0) = 0;
                tensor::Tensor input = model_.fill_input(pos_tensor, embedding, true);
                int32_t next = -1;
                model_.predict(input, pos_tensor, true, next);
            }
            
            std::cout << "迭代 " << (iteration + 1) << ":\n";
            std::cout << "  CPU变化: " << get_cpu_memory_change_mb() << " MB\n";
            std::cout << "  GPU变化: " << get_gpu_memory_change_mb() << " MB\n";
        }
        
        double cpu_memory_growth = static_cast<double>(get_cpu_memory_change_mb());
        double gpu_memory_growth = static_cast<double>(get_gpu_memory_change_mb());
        
        std::cout << "内存泄漏检测结果:\n";
        std::cout << "  CPU内存变化: " << cpu_memory_growth << " MB\n";
        std::cout << "  GPU内存变化: " << gpu_memory_growth << " MB\n";
        
        if (cpu_memory_growth > 100 || gpu_memory_growth > 100) {
            std::cout << "⚠️  可能存在内存泄漏\n";
        } else if (cpu_memory_growth > 20 || gpu_memory_growth > 20) {
            std::cout << "⚠️  内存增长较多，需要关注\n";
        } else {
            std::cout << "✅ 内存使用正常\n";
        }
    }

private:
    void generate_sequence(const std::string& prompt, int target_length) {
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
        
        // 生成tokens直到达到目标长度
        int32_t current = tokens.back();
        int current_length = static_cast<int>(tokens.size());
        
        while (current_length < target_length) {
            std::vector<int32_t> tokens_vec = {current};
            auto embedding = model_.embedding(tokens_vec);
            pos_tensor.index<int32_t>(0) = current_length - 1;
            tensor::Tensor input_tensor = model_.fill_input(pos_tensor, embedding, false);
            
            int32_t next = -1;
            model_.predict(input_tensor, pos_tensor, false, next);
            
            if (model_.is_sentence_ending(next)) {
                break;
            }
            
            current = next;
            current_length++;
        }
    }

};

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        LOG(INFO) << "Usage: ./memory_benchmark checkpoint_path tokenizer_path [--quant]";
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
    
    // 获取模型初始化前的GPU内存使用
    size_t gpu_memory_before = 0;
    size_t free_memory, total_memory;
    cudaError_t result = cudaMemGetInfo(&free_memory, &total_memory);
    if (result == cudaSuccess) {
        gpu_memory_before = (total_memory - free_memory) / (1024 * 1024);
        std::cout << "模型初始化前GPU内存使用: " << gpu_memory_before << " MB\n";
    }
    
    // 初始化模型
    model::LLama3Model model(base::TokenizerType::kEncodeBpe , tokenizer_path,
                             checkpoint_path, use_quantization);
    
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "模型初始化失败: " << init_status.get_err_code();
        return -1;
    }
    
    // 获取模型初始化后的GPU内存使用
    size_t gpu_memory_after = 0;
    int64_t model_gpu_memory = 0; // 声明在外部作用域
    result = cudaMemGetInfo(&free_memory, &total_memory);
    if (result == cudaSuccess) {
        gpu_memory_after = (total_memory - free_memory) / (1024 * 1024);
        std::cout << "模型初始化后GPU内存使用: " << gpu_memory_after << " MB\n";
        // 使用有符号整数避免下溢
        model_gpu_memory = static_cast<int64_t>(gpu_memory_after) - static_cast<int64_t>(gpu_memory_before);
        std::cout << "模型占用GPU内存: " << model_gpu_memory << " MB\n";
    }
    
    MemoryBenchmark benchmark(model);
    benchmark.set_model_gpu_memory(static_cast<size_t>(std::max<int64_t>(0, model_gpu_memory))); // 使用计算出的模型GPU内存
    
    std::cout << "=== LLaMA3 内存使用测试 ===\n";
    std::cout << "模型类型: " << (use_quantization ? "量化模型" : "标准模型") << "\n";
    
    // 1. 基础内存使用
    benchmark.test_base_memory_usage();
    
    // 2. 序列长度对内存的影响
    benchmark.test_sequence_length_memory();
    
    // 3. KV Cache增长测试  
    benchmark.test_kv_cache_growth();
    
    // 4.内存泄漏检测
    benchmark.test_memory_leak();
    
    std::cout << "\n内存测试完成！\n";
    return 0;
} 