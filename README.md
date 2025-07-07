# KuiperInfer - High-Performance LLM Inference Framework

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.19+-orange.svg)](https://cmake.org/)
[![Stars](https://img.shields.io/github/stars/your-username/KuiperInfer?style=social)](https://github.com/your-username/KuiperInfer)

> A high-performance, production-ready inference framework for Large Language Models (LLMs) with support for Llama2/3 and Qwen2.5 series models.

## 🚀 Features

- **Multi-Model Support**: Native support for Llama2, Llama3.2, Qwen2.5, and Qwen3 models
- **High Performance**: Optimized CUDA kernels with CPU fallback for maximum throughput
- **Quantization**: INT8 quantization support for reduced memory footprint and faster inference
- **Modern C++**: Built with C++17/20 standards for type safety and performance
- **Production Ready**: Comprehensive error handling, logging, and testing infrastructure
- **Easy Integration**: Simple API for model loading and text generation
- **Cross-Platform**: Support for Linux with CUDA acceleration

## 📊 Performance

| Model | Precision | Platform | Speed (tokens/s) | Memory Usage |
|-------|-----------|----------|------------------|--------------|
| Llama2-7B | FP32 | RTX 3060 | ~60 | ~14GB |
| Llama2-7B | INT8 | RTX 3060 | ~80 | ~7GB |
| Llama3.2-1B | FP32 | RTX 3060 | ~120 | ~2GB |
| Qwen2.5-0.5B | FP32 | RTX 3060 | ~200 | ~1GB |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Layer   │    │  Operator Layer │    │  Kernel Layer   │
│                 │    │                 │    │                 │
│ • Llama3Model   │───▶│ • Layer         │───▶│ • CPU Kernels   │
│ • QwenModel     │    │ • MatMul        │    │ • CUDA Kernels  │
│ • Model Base    │    │ • MHA           │    │ • Memory Mgmt   │
└─────────────────┘    │ • RMSNorm       │    └─────────────────┘
                       │ • SwiGLU        │
                       └─────────────────┘
```

## ⚡ Technical Highlights

### 🎯 **Optimized CUDA Kernels**
- Hand-crafted CUDA kernels for maximum performance
- Optimized memory access patterns and thread configurations
- Support for mixed precision (FP16/FP32) operations

### 🧠 **Advanced Quantization**
- INT8 quantization with minimal accuracy loss
- Dynamic quantization for flexible memory usage
- Quantization-aware training support

### 🏭 **Production-Ready Design**
- Modern C++17/20 with RAII and smart pointers
- Comprehensive error handling and logging
- Extensive unit tests and benchmarks
- CMake-based build system with dependency management

### 🔧 **Developer Experience**
- Clean, modular architecture
- Comprehensive documentation
- Easy model conversion from Hugging Face
- Interactive demos and examples

## 🛠️ Installation

### Prerequisites

- **CUDA Toolkit** 12.2 or later
- **CMake** 3.19 or later
- **GCC** 9.0 or later (with C++17 support)
- **Python** 3.8+ (for model export tools)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/KuiperInfer.git
cd KuiperInfer

# Create build directory
mkdir build && cd build

# Configure with automatic dependency download
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON -DQWEN2_SUPPORT=ON ..

# Build the project
make -j$(nproc)
```

### Manual Dependencies (Optional)

If you prefer to install dependencies manually:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install libglog-dev libgtest-dev libsentencepiece-dev \
                     libarmadillo-dev libopenblas-dev

# Install CUDA Toolkit
# Follow NVIDIA's official installation guide
```

## 📖 Usage

### Model Download

Download pre-trained models from Hugging Face:

```bash
# Set Hugging Face mirror for faster download (optional)
export HF_ENDPOINT=https://hf-mirror.com

# Download Llama2 model
huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf \
    --local-dir meta-llama/Llama-2-7b-hf --local-dir-use-symlinks False

# Download Llama3.2 model
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B \
    --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False

# Download Qwen2.5 model
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B \
    --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```

### Model Export

Convert Hugging Face models to our optimized binary format:

```bash
# Export Llama2 model
python3 tools/export.py llama2_7b.bin --hf=meta-llama/Llama-2-7b-hf

# Export Llama3.2 model
python3 tools/export.py llama3.2_1b.bin --hf=meta-llama/Llama-3.2-1B

# Export Qwen2.5 model
python3 tools/export_qwen2.py qwen2.5_0.5b.bin --hf=Qwen/Qwen2.5-0.5B

# Export with INT8 quantization
python3 tools/export.py llama2_7b_int8.bin --hf=meta-llama/Llama-2-7b-hf --version3
```

### Text Generation

Run inference with different models:

```bash
# Llama2 inference
./build/demo/llama_infer llama2_7b.bin tokenizer.model

# Llama3.2 inference
./build/demo/llama_infer llama3.2_1b.bin meta-llama/Llama-3.2-1B/tokenizer.json

# Qwen2.5 inference
./build/demo/qwen_infer qwen2.5_0.5b.bin Qwen/Qwen2.5-0.5B/tokenizer.json

# Compare with Hugging Face results
python3 hf_infer/llama3_infer.py
python3 hf_infer/qwen2_infer.py
```

### Interactive Chat Demo

```bash
# Start interactive chat with Llama3.2
./build/demo/chat_qwen

# Start interactive chat with Qwen2.5
./build/demo/chat_qwen
```

### Programmatic Usage

```cpp
#include "model/llama3.h"

// Initialize model
auto model = std::make_unique<model::Llama3Model>(
    base::TokenizerType::kEncodeBpe,
    "tokenizer.json",
    "llama3.2_1b.bin",
    false  // is_quantized
);

// Initialize on CUDA
model->init(base::DeviceType::kDeviceCUDA);

// Generate text
std::vector<int> tokens = model->encode("Hello, world!");
int next_token;
model->predict(input_tensor, pos_tensor, true, next_token);
```

## 🔧 Configuration

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `USE_CPM` | Use CPM for automatic dependency management | OFF |
| `LLAMA3_SUPPORT` | Enable Llama3 model support | OFF |
| `QWEN2_SUPPORT` | Enable Qwen2.5 model support | OFF |
| `QWEN3_SUPPORT` | Enable Qwen3 model support | OFF |
| `BUILD_TESTS` | Build test suite | ON |
| `BUILD_DEMO` | Build demo applications | ON |

### Runtime Configuration

```cpp
// Configure CUDA settings
auto cuda_config = std::make_shared<kernel::CudaConfig>();
cuda_config->set_stream(cuda_stream);
cuda_config->set_memory_pool(memory_pool);

// Set model parameters
model->set_max_seq_len(2048);
model->set_temperature(0.7);
model->set_top_p(0.9);
```

## 🧪 Testing

```bash
# Run all tests
cd build
make test

# Run specific test suites
./test/test_model/test_llama_cpu
./test/test_op/test_cu_matmul
./test/test_tensor/test_tensor
```

## 📈 Benchmarks

Run performance benchmarks:

```bash
# CPU benchmarks
./build/test/benchmark_cpu

# CUDA benchmarks
./build/test/benchmark_cuda

# Memory usage analysis
./build/test/memory_benchmark
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
sudo apt-get install clang-format clang-tidy cppcheck

# Setup pre-commit hooks
pre-commit install

# Run code formatting
make format

# Run static analysis
make static-analysis
```

### Code Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use meaningful variable and function names
- Add comprehensive unit tests for new features
- Document public APIs with Doxygen comments

## 📚 Documentation

- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Performance Tuning](docs/performance.md)
- [Model Conversion](docs/model_conversion.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🏆 Acknowledgments

- [Meta AI](https://ai.meta.com/) for Llama models
- [Alibaba Cloud](https://www.alibabacloud.com/) for Qwen models
- [NVIDIA](https://developer.nvidia.com/) for CUDA toolkit
- [Google](https://github.com/google) for glog, gtest, and sentencepiece

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Course Materials](https://tvle9mq8jh.feishu.cn/docx/AGb0dpqwfohQ9oxx4QycqbCjnJh)
- [Issues](https://github.com/your-username/KuiperInfer/issues)
- [Discussions](https://github.com/your-username/KuiperInfer/discussions)
- [Releases](https://github.com/your-username/KuiperInfer/releases)

## 📞 Contact & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/KuiperInfer/issues)
- **Discussions**: [Join the community](https://github.com/your-username/KuiperInfer/discussions)
- **Email**: contact@kuiperinfer.com
- **WeChat**: lyrry1997 (for course inquiries)
- **Course**: [动手自制大模型推理框架](https://tvle9mq8jh.feishu.cn/docx/AGb0dpqwfohQ9oxx4QycqbCjnJh)

## 🎓 Learning Resources

This project is part of the **"Hands-on LLM Inference Framework"** course, which covers:
- Modern C++ development practices
- CUDA programming and optimization
- LLM architecture and implementation
- Production-ready software engineering

For course enrollment and detailed learning materials, please contact us via WeChat or visit the course page.

---

<div align="center">
  <p>Made with ❤️ by the KuiperInfer Team</p>
  <p>If you find this project helpful, please give it a ⭐️</p>
</div>