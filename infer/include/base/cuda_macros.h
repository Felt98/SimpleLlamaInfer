#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

#include <cuda_runtime.h>

// CUDA基础常量
#define WARP_SIZE 32

// 数据类型转换宏 - 支持const和非const
#define CONST_FLOAT4(ptr, idx) (reinterpret_cast<const float4*>(ptr)[(idx)])
#define FLOAT4_LOAD(ptr, idx) (reinterpret_cast<const float4*>(ptr)[(idx)])
#define FLOAT4_STORE(ptr, idx, val) (reinterpret_cast<float4*>(ptr)[(idx)] = (val))

#define CONST_INT4(ptr, idx) (reinterpret_cast<const int4*>(ptr)[(idx)])
#define INT4_LOAD(ptr, idx) (reinterpret_cast<const int4*>(ptr)[(idx)])
#define INT4_STORE(ptr, idx, val) (reinterpret_cast<int4*>(ptr)[(idx)] = (val))

#define CONST_HALF2(ptr, idx) (reinterpret_cast<const half2*>(ptr)[(idx)])
#define HALF2_LOAD(ptr, idx) (reinterpret_cast<const half2*>(ptr)[(idx)])
#define HALF2_STORE(ptr, idx, val) (reinterpret_cast<half2*>(ptr)[(idx)] = (val))

// 保留原有宏（用于非const情况）
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// CUDA线程索引宏
#define THREAD_IDX (threadIdx.x + blockIdx.x * blockDim.x)
#define THREAD_IDX_2D (threadIdx.x + blockIdx.x * blockDim.x + \
                       threadIdx.y + blockIdx.y * blockDim.y * gridDim.x * blockDim.x)

// 内存访问宏
#define GLOBAL_MEM_ALIGN 256
#define SHARED_MEM_ALIGN 128

// 同步宏
#define WARP_SYNC() __syncwarp()
#define BLOCK_SYNC() __syncthreads()

// 数学运算宏
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ABS(x) ((x) < 0 ? -(x) : (x))

// 位操作宏
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))

// 向量化访问宏
#define VECTORIZED_LOAD(ptr, idx) FLOAT4((ptr)[(idx)])
#define VECTORIZED_STORE(ptr, idx, value) ((ptr)[(idx)] = FLOAT4(value))

// 线程块配置宏
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_WARP_SIZE 32
#define DEFAULT_GRID_SIZE(block_size, total_size) \
    ((total_size + block_size - 1) / block_size)

// 内存拷贝宏
#define MEMCPY_128BITS(dst, src) \
    do { \
        FLOAT4(dst) = FLOAT4(src); \
    } while(0)

// 原子操作宏
#define ATOMIC_ADD_FLOAT(addr, val) \
    atomicAdd((float*)(addr), (float)(val))

#define ATOMIC_MAX_FLOAT(addr, val) \
    atomicMax((int*)(addr), __float_as_int(val))

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

// 调试宏
#ifdef DEBUG
#define CUDA_DEBUG_PRINT(fmt, ...) \
    printf("[CUDA DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define CUDA_DEBUG_PRINT(fmt, ...) do {} while(0)
#endif

// 性能优化宏
#define UNROLL_LOOP(n) _Pragma("unroll")
#define NO_UNROLL_LOOP() _Pragma("nounroll")

// 内存对齐宏
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(x, align) ((x) & ~((align) - 1))

// 向量化循环宏
#define VECTORIZED_LOOP(start, end, step) \
    for (int i = (start); i < (end); i += (step))

// 共享内存访问宏
#define SHARED_LOAD(shared_ptr, idx) (shared_ptr)[(idx)]
#define SHARED_STORE(shared_ptr, idx, value) ((shared_ptr)[(idx)] = (value))

#endif // CUDA_MACROS_H 