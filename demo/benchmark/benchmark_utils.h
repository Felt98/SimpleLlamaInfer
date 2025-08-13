#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <chrono>
#include <string>
#include <vector>

// 时间测量工具函数
inline double duration_ms(const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double duration_seconds(const std::chrono::high_resolution_clock::time_point& start,
                              const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

// 内存监控工具（简化版）
inline size_t get_memory_usage() {
    // 这里可以添加实际的内存监控代码
    // 例如通过/proc/self/status或CUDA内存API
    return 0; // 占位符
}

// 统计计算工具
template<typename T>
std::pair<T, T> calculate_mean_std(const std::vector<T>& values) {
    if (values.empty()) return {0, 0};
    
    T mean = 0;
    for (const auto& val : values) {
        mean += val;
    }
    mean /= values.size();
    
    T variance = 0;
    for (const auto& val : values) {
        T diff = val - mean;
        variance += diff * diff;
    }
    variance /= values.size();
    
    return {mean, std::sqrt(variance)};
}

// 输出格式化工具
void print_performance_header();
void print_performance_footer();
void save_results_to_csv(const std::string& filename, 
                        const std::vector<std::string>& test_names,
                        const std::vector<double>& latencies,
                        const std::vector<double>& throughputs);

#endif // BENCHMARK_UTILS_H 