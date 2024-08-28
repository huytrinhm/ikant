#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <initializer_list>
#include <iostream>

namespace KAN {

struct Tensor {
  uint32_t dim;
  uint32_t* shape;
  uint32_t* stride;
  float* data;

  Tensor(uint32_t dim = 0,
         uint32_t* shape = nullptr,
         uint32_t* stride = nullptr,
         float* data = nullptr);

  Tensor(std::initializer_list<uint32_t> shape_init, float* data);

  Tensor(std::initializer_list<uint32_t> shape_init);

  void print_tensor_data(std::ostream& os,
                         uint32_t _dim,
                         uint32_t* shape,
                         uint32_t* stride,
                         float* data) const;

  float& operator()(uint32_t i0...);
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

Tensor tensor_from_file(const char* filename);

void tensor_to_file(Tensor& tensor, const char* filename);

struct GC_Tensor : Tensor {
  using Tensor::Tensor;

  ~GC_Tensor();

  GC_Tensor(Tensor&& other);

  GC_Tensor& operator=(Tensor&& other);

  GC_Tensor(const GC_Tensor&) = delete;
  GC_Tensor(GC_Tensor&&) = delete;
  GC_Tensor& operator=(const GC_Tensor&) = delete;
  GC_Tensor& operator=(GC_Tensor&&) = delete;
};

}  // namespace KAN
#endif