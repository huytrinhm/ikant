#ifndef TENSOR_H
#define TENSOR_H

#include <cstdarg>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>

namespace KAN {

struct Tensor {
  uint32_t dim;
  uint32_t* shape;
  uint32_t* stride;
  float* data;

  Tensor(uint32_t dim = 0,
         uint32_t* shape = nullptr,
         uint32_t* stride = nullptr,
         float* data = nullptr)
      : dim(dim), shape(shape), stride(stride), data(data) {
    if (!stride && shape) {
      stride = new uint32_t[dim];
      stride[dim - 1] = 1;
      for (int32_t i = dim - 2; i >= 0; --i)
        stride[i] = stride[i + 1] * shape[i + 1];
    }
  }

  // ~Tensor() {
  //     delete[] shape;
  //     delete[] stride;
  // }

  Tensor(std::initializer_list<uint32_t> shape_init, float* data) : data(data) {
    dim = shape_init.size();

    shape = new uint32_t[dim];
    std::copy(shape_init.begin(), shape_init.end(), shape);

    stride = new uint32_t[dim];
    stride[dim - 1] = 1;
    for (int32_t i = dim - 2; i >= 0; --i)
      stride[i] = stride[i + 1] * shape[i + 1];
  }

  Tensor(std::initializer_list<uint32_t> shape_init) {
    dim = shape_init.size();

    shape = new uint32_t[dim];
    std::copy(shape_init.begin(), shape_init.end(), shape);

    stride = new uint32_t[dim];
    stride[dim - 1] = 1;
    for (int32_t i = dim - 2; i >= 0; --i)
      stride[i] = stride[i + 1] * shape[i + 1];

    uint32_t size = 1;
    for (uint32_t i = 0; i < dim; ++i)
      size *= shape[i];

    data = new float[size];
  }

  void print_tensor_data(std::ostream& os,
                         uint32_t _dim,
                         uint32_t* shape,
                         uint32_t* stride,
                         float* data) const {
    if (dim == 0) {
      std::cerr << "Tried to access uninitialized tensor." << std::endl;
      throw;
    }

    std::string indent = std::string(2 * (dim - _dim), ' ');
    os << indent << "[";
    if (_dim == 1) {
      for (uint32_t i = 0; i < *shape - 1; ++i)
        os << std::format("{: .4f}", data[i]) << ", ";
      os << std::format("{: .4f}", data[*shape - 1]) << "]";
      return;
    }

    os << "\n";

    for (uint32_t i = 0; i < *shape - 1; ++i) {
      print_tensor_data(os, _dim - 1, shape + 1, stride + 1, data);
      os << ",\n";
      data += *stride;
    }

    print_tensor_data(os, _dim - 1, shape + 1, stride + 1, data);
    os << "\n" << indent << "]";
  }

  float& operator()(uint32_t i0...) {
    if (dim < 1) {
      std::cerr << "Tried to access uninitialized tensor." << std::endl;
      throw;
    }

    float* result = data;
    result += i0 * stride[0];

    va_list args;
    va_start(args, i0);

    for (uint32_t count = 1; count < dim; count++) {
      result += va_arg(args, uint32_t) * stride[count];
    }

    va_end(args);

    return *result;
  }
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  tensor.print_tensor_data(os, tensor.dim, tensor.shape, tensor.stride,
                           tensor.data);
  return os;
}

Tensor tensor_from_file(const char* filename) {
  std::ifstream fin(filename, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw;
  }

  Tensor result;
  fin.read(reinterpret_cast<char*>(&result.dim), sizeof(uint32_t));

  result.shape = new uint32_t[result.dim];
  result.stride = new uint32_t[result.dim];

  fin.read(reinterpret_cast<char*>(result.shape),
           result.dim * sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(result.stride),
           result.dim * sizeof(uint32_t));

  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < result.dim; ++i) {
    num_elements *= result.shape[i];
  }

  result.data = new float[num_elements];
  fin.read(reinterpret_cast<char*>(result.data), num_elements * sizeof(float));

  fin.close();

  return result;
}

void tensor_to_file(Tensor& tensor, const char* filename) {
  std::ofstream fout(filename, std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw;
  }

  fout.write(reinterpret_cast<char*>(&tensor.dim), sizeof(uint32_t));

  fout.write(reinterpret_cast<char*>(tensor.shape),
             tensor.dim * sizeof(uint32_t));
  fout.write(reinterpret_cast<char*>(tensor.stride),
             tensor.dim * sizeof(uint32_t));

  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < tensor.dim; ++i) {
    num_elements *= tensor.shape[i];
  }

  fout.write(reinterpret_cast<char*>(tensor.data),
             num_elements * sizeof(float));

  fout.close();
}

struct GC_Tensor : Tensor {
  using Tensor::Tensor;

  ~GC_Tensor() {
    delete[] shape;
    delete[] stride;
    delete[] data;
  }

  GC_Tensor(Tensor&& other) : Tensor(other) {}

  GC_Tensor& operator=(Tensor&& other) {
    delete[] shape;
    delete[] stride;
    delete[] data;

    dim = other.dim;
    shape = other.shape;
    stride = other.stride;
    data = other.data;
    return *this;
  }

  GC_Tensor(const GC_Tensor&) = delete;
  GC_Tensor(GC_Tensor&&) = delete;
  GC_Tensor& operator=(const GC_Tensor&) = delete;
  GC_Tensor& operator=(GC_Tensor&&) = delete;
};

}  // namespace KAN
#endif