# SimpleLlamaInfer - Simple and Lightweight LLM Inference Framework

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.19+-orange.svg)](https://cmake.org/)
[![Stars](https://img.shields.io/github/stars/your-username/SimpleLlamaInfer?style=social)](https://github.com/your-username/SimpleLlamaInfer)

> A simple and lightweight inference engine for Large Language Models (LLMs) with support for Llama3.2 models.

## 🚀 Features

- **Llama3 models Support**: Native support for Llama3.2 models
- **High Performance**: Optimized CUDA kernels with CPU fallback for maximum throughput
- **Quantization**: INT8 quantization support for reduced memory footprint and faster inference
- **Modern C++**: Built with C++17/20 standards for type safety and performance
- **Production Ready**: Comprehensive error handling, logging, and testing infrastructure
- **Easy Integration**: Simple API for model loading and text generation
- **Cross-Platform**: Support for Linux with CUDA acceleration



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

- **CUDA Toolkit** 11.8 or later (tested with CUDA 11.8.89)
- **CMake** 3.19 or later
- **GCC** 9.0 or later (with C++17 support)
- **Python** 3.8+ (for model export tools)
- **Conda** (for dependency management)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/SimpleLlamaInfer.git
cd SimpleLlamaInfer

# Create build directory
mkdir build && cd build

# Configure with automatic dependency download
cmake ..

# Build the project
make -j$(nproc)
```

### Required Dependencies

| Library | Version | Description |
|---------|---------|-------------|
| **CUDA Toolkit** | 11.8+ | NVIDIA CUDA for GPU acceleration |
| **sentencepiece** | 0.2.0 | Text tokenization library |
| **glog** | 0.6.0 | Google logging library |
| **gtest** | 1.17.0 | Google Test framework |
| **armadillo** | 14.6.2 | Linear algebra library |
| **re2** | 2023.03.02 | Regular expression library |
| **nlohmann_json** | 3.12.0 | JSON parsing library |
| **libabseil** | 20240116.1 | Google Abseil C++ library |
| **gflags** | 2.2.2 | Command line flags library |

### Manual Dependencies (Optional)

If you prefer to install dependencies manually:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install \
    libsentencepiece-dev \
    libglog-dev \
    libgtest-dev \
    libarmadillo-dev \
    libre2-dev \
    nlohmann-json3-dev \
    libgflags-dev

# Install CUDA Toolkit
# Follow NVIDIA's official installation guide
```

## 📖 Usage

### Model Download

Download pre-trained Llama3.2 models from Hugging Face:

- **Llama3.2-1B**: https://huggingface.co/meta-llama/Llama-3.2-1B
- **Llama3.2-3B**: https://huggingface.co/meta-llama/Llama-3.2-3B

You can download these models using the Hugging Face CLI or directly from the web interface.


### Model Export

Convert Hugging Face models to our optimized binary format:

```bash
# Export Llama3.2 model
python3 tools/export_llama.py --version 0 --hf meta-llama/Llama-3.2-1B llama3.2_1b.bin

# Export INT8 quantized Llama3.2 models
python3 tools/export_llama.py --version 3 --hf meta-llama/Llama-3.2-1B llama3.2_1b_int8.bin
```


### Text Generation

Run inference with different models:

```bash
# Llama3.2 inference
./build/examples/llama_infer llama3.2_1b.bin meta-llama/Llama-3.2-1B/tokenizer.json

# Quantized Llama3.2 inference
./build/examples/llama_infer llama3.2_1b_int8.bin meta-llama/Llama-3.2-1B/tokenizer.json --quant

# Compare with Hugging Face results
python3 hf_infer/llama3_infer.py
```

## 🧪 Testing

```bash
# Run all tests
cd build
make test

# Run specific test suites
./test/test_model/test_llama3_cpu
./test/test_op/test_cu_matmul
./test/test_tensor/test_tensor
```

## 📈 Benchmarks

Run performance benchmarks:
```bash
# Performance test for FP32 model
./build/benchmarks/performance_test llama3.2_1b.bin meta-llama/Llama-3.2-1B/tokenizer.json

# Performance test for INT8 quantized model
./build/benchmarks/performance_test llama3.2_1b_int8.bin meta-llama/Llama-3.2-1B/tokenizer.json --quant

# Latency benchmark for FP32 model
./build/benchmarks/latency_benchmark llama3.2_1b.bin meta-llama/Llama-3.2-1B/tokenizer.json

# Latency benchmark for INT8 quantized model
./build/benchmarks/latency_benchmark llama3.2_1b_int8.bin meta-llama/Llama-3.2-1B/tokenizer.json --quant

# Compare accuracy between FP32 and INT8 models
./build/benchmarks/accuracy_benchmark llama3.2_1b.bin llama3.2_1b_int8.bin meta-llama/Llama-3.2-1B/tokenizer.json
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

## 🏆 Acknowledgments

### Reference

This project is inspired by and references the following excellent open-source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - High-performance LLM inference in C/C++
- [llama3.cpp](https://github.com/jonemeth/llama3.cpp) - Llama3 inference implementation
- [LlamaInfer](https://github.com/AllenJWZhu/LlamaInfer) - Llama model inference framework
- [LLM-InferenceNet](https://github.com/adithya-s-k/LLM-InferenceNet) - LLM inference optimization

### Dependencies

- [Meta AI](https://ai.meta.com/) for Llama3.2 models
- [NVIDIA](https://developer.nvidia.com/) for CUDA toolkit
- [Google](https://github.com/google) for glog, gtest, and sentencepiece

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.