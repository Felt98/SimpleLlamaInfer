#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  // 使用Armadillo库创建浮点向量，从input1张量的数据指针构造
  // const_cast<float*> 用于移除const限定符，因为Armadillo需要非const指针
  // input1.size() 指定向量大小
  // false 表示不复制数据（使用外部内存）
  // true 表示严格模式（检查内存对齐等）
  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
  // Armadillo 的语法：将input_vec1和input_vec2相加，结果存储在output_vec中
  output_vec = input_vec1 + input_vec2;
}

}  // namespace kernel
