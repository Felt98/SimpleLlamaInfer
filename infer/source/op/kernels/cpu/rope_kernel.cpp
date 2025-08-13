#include "rope_kernel.h"
namespace kernel {
#if defined (LLAMA3_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim ++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec =
            const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                      : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#elif defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim ++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec =
            const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                      : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#else
// 计算并缓存所有位置（pos）和所有 head 维度（head_dim）下的正弦（sin）和余弦（cos）值，供后续 RoPE 旋转使用
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));  // 计算频率10000^(−i/d)
      float val = static_cast<float>(pos) * freq;   // 当前位置 pos 对应的旋转角度θ=pos*freq
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;     // sin_cache[pos][head_dim]
      *(cos_cache + pos * head_size + head_dim) = fcr;     // cos_cache[pos][head_dim]
    }
  }
}

// 对输入当前 token的 query（input_q）和 key（input_k）向量进行 RoPE 旋转位置编码
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += 2) {  // 遍历所有维度，对偶数/奇数维度成对旋转
    int32_t head_dim = i % head_size;  // 当前维度在 head 内的下标
    float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);  // sin_cache[pos][head_dim]
    float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);  // cos_cache[pos][head_dim]

    // 决定对 query 和 key 都做旋转（i < kv_dim 时），还是只对 query 做（i >= kv_dim 时）
    int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only

    for (int32_t v = 0; v < rotn; v++) {
      float* vec =
          const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                    : input_k.ptr<float>());  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];

      // x'=x⋅cos(θ)−y⋅sin(θ)
      // y'=x⋅sin(θ)+y⋅cos(θ)
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
    /*
    // 对 query 做旋转
    float* vec_q=const_cast<float*>(input_q.ptr<float>());
    float v0 = vec_q[i];
    float v1 = vec_q[i + 1];
    vec_q[i] = v0 * fcr - v1 * fci;
    vec_q[i + 1] = v0 * fci + v1 * fcr;
    // 如果 i < kv_dim，则对 key 也做旋转
    if(i<kv_dim){
      float* vec_k=const_cast<float*>(input_k.ptr<float>());
      float v0 = vec_k[i];
      float v1 = vec_k[i + 1];
      vec_k[i] = v0 * fcr - v1 * fci;
      vec_k[i + 1] = v0 * fci + v1 * fcr;
    }
    */
  }
}
#endif
}  // namespace kernel