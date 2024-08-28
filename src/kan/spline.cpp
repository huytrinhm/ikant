#include "spline.h"
#include <cstdint>
#include "tensor.h"

namespace KAN {

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

}  // namespace KAN