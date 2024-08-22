#ifndef SPLINE_H
#define SPLINE_H

#include <cstdint>
#include <cstring>
#include "tensor.h"

namespace KAN {

/*
Compute the B-spline bases for the given sample.

Args:
    grid (Tensor): Grid tensor of shape (in_features, grid_size + 2 *
spline_order + 1). x (Tensor): Input tensor of shape (in_features). spline_order
(int): Spline order.

Returns:
    Tensor: B-spline bases tensor of shape (in_features, grid_size +
spline_order).
*/
void b_splines(Tensor& grid,
               Tensor& x,
               uint32_t spline_order,
               Tensor* bases,
               Tensor* bases_temp,
               Tensor* bases_minus_1) {
  uint32_t in_features = x.shape[0];
  uint32_t num_grid_points = grid.shape[0];
  uint32_t num_bases = num_grid_points - spline_order - 1;

  Tensor* prev_bases = bases_temp;
  Tensor* current_bases = bases_temp;

  for (uint32_t j = 0; j < in_features; ++j)
    for (uint32_t t = 0; t < num_grid_points - 1; ++t)
      (*current_bases)(j, t) = (x(j) >= grid(t)) && (x(j) < grid(t + 1));

  for (uint32_t k = 1; k <= spline_order; ++k) {
    if ((k == spline_order - 1) && bases_minus_1) {
      prev_bases = current_bases;
      current_bases = bases_minus_1;
    } else if (k == spline_order) {
      prev_bases = current_bases;
      current_bases = bases;
    }

    for (uint32_t j = 0; j < in_features; ++j)
      for (int32_t t = 0; t < num_grid_points - k - 1; ++t)
        (*current_bases)(j, t) =
            (((x(j) - grid(t)) / (grid(t + k) - grid(t)) *
              (*prev_bases)(j, t)) +
             ((grid(t + k + 1) - x(j)) / (grid(t + k + 1) - grid(t + 1)) *
              (*prev_bases)(j, t + 1)));
  }
}

void b_splines_derivative(Tensor& grid,
                          Tensor& bases_minus_1,
                          Tensor& coeff,
                          Tensor& partial_grad,
                          Tensor& spline_weights,
                          uint32_t spline_order,
                          Tensor& grad) {
  uint32_t in_features = bases_minus_1.shape[0];
  uint32_t out_features = coeff.shape[0];
  uint32_t num_grid_points = grid.shape[0];
  uint32_t num_bases = num_grid_points - spline_order;

  for (uint32_t j = 0; j < in_features; ++j) {
    for (uint32_t k = 0; k < out_features; ++k) {
      float accumulator = 0;
      for (uint32_t t = 0; t < num_bases; ++t)
        accumulator +=
            coeff(k, j, t) *
            (bases_minus_1(j, t) / (grid(t + spline_order) - grid(t)) -
             bases_minus_1(j, t + 1) /
                 (grid(t + spline_order + 1) - grid(t + 1)));
      grad(j) =
          spline_order * accumulator * partial_grad(k) * spline_weights(k, j);
    }
  }
}

Tensor test_b_splines(Tensor& grid,
                      Tensor& Xs,
                      uint32_t spline_order,
                      Tensor& coeff) {
  if (spline_order <= 1) {
    std::cerr << "spline_order must larger than 1." << std::endl;
    throw;
  }

  uint32_t num_samples = Xs.shape[0];
  uint32_t in_features = Xs.shape[1];
  uint32_t num_grid_points = grid.shape[1];
  uint32_t num_bases = num_grid_points - spline_order - 1;

  float* bases_temp_data = new float[in_features * (num_grid_points - 1)];
  float* bases_minus_1_data =
      new float[in_features * (num_grid_points - spline_order)];
  float* bases_data =
      new float[in_features * (num_grid_points - spline_order - 1)];
  float* grad_data = new float[in_features];

  memset(grad_data, 0, in_features * sizeof(float));

  Tensor bases_temp({in_features, num_grid_points - 1}, bases_temp_data);
  Tensor bases_minus_1({in_features, num_grid_points - spline_order},
                       bases_minus_1_data);
  Tensor bases({in_features, num_grid_points - spline_order - 1}, bases_data);
  Tensor grad({in_features}, grad_data);

  for (uint32_t i = 0; i < num_samples; ++i) {
    Tensor x({in_features}, &Xs(i, 0));

    b_splines(grid, x, spline_order, &bases, &bases_temp, &bases_minus_1);
    // b_splines_derivative(grid, bases_minus_1, coeff, spline_order, grad);
  }

  return grad;
}

}  // namespace KAN
#endif