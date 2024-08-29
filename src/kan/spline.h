#ifndef SPLINE_H
#define SPLINE_H

#include <cstdint>
#include "tensor.h"

namespace KAN {

/*
Compute the B-spline bases for the given sample.

Args:
  - grid (Tensor): Grid tensor of shape (grid_size + 2 * spline_order + 1).
  - x (Tensor): Input tensor of shape (in_features).
  - spline_order (int): Spline order.

Returns:
  - Tensor: B-spline bases tensor of shape (in_features, grid_size +
spline_order).
*/
void b_splines(Tensor& grid,
               Tensor& x,
               uint32_t spline_order,
               Tensor* bases,
               Tensor* bases_temp,
               Tensor* bases_minus_1);

void b_splines_derivative(Tensor& grid,
                          Tensor& bases_minus_1,
                          Tensor& coeff,
                          Tensor& partial_grad,
                          Tensor& spline_weights,
                          uint32_t spline_order,
                          Tensor& grad);

}  // namespace KAN
#endif