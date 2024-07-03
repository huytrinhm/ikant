#ifndef SPLINE_H
#define SPLINE_H

#include "tensor.h"

namespace KAN {

/*
Compute the B-spline bases for the given input tensor.

Args:
    grid (Tensor): Grid tensor of shape (in_features, grid_size + 2 * spline_order + 1).
    x (Tensor): Input tensor of shape (batch_size, in_features).
    spline_order (int): Spline order.

Returns:
    Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
*/
void b_splines(Tensor& grid, Tensor& x, uint32_t spline_order, Tensor* bases, Tensor* bases_temp, Tensor* bases_minus_1) {
	uint32_t num_samples = x.shape[0];
	uint32_t in_features = x.shape[1];
	uint32_t num_grid_points = grid.shape[1];
	uint32_t num_bases = num_grid_points - spline_order - 1;

	Tensor* prev_bases = bases_temp;
	Tensor* current_bases = bases_temp;

	for (uint32_t i = 0; i < num_samples; ++i)
		for (uint32_t j = 0; j < in_features; ++j)
			for (uint32_t t = 0; t < num_grid_points - 1; ++t)
				(*current_bases)(i, j, t) = (x(i, j) >= grid(j, t)) && (x(i, j) < grid(j, t + 1));

	for (uint32_t k = 1; k <= spline_order; ++k) {
		if ((k == spline_order - 1) && bases_minus_1) {		
			prev_bases = current_bases;
			current_bases = bases_minus_1;
		} else if (k == spline_order) {
			prev_bases = current_bases;
			current_bases = bases;
		}

		for (uint32_t i = 0; i < num_samples; ++i)
			for (uint32_t j = 0; j < in_features; ++j)
				for (int32_t t = 0; t <= num_grid_points - k - 1; ++t)
					(*current_bases)(i, j, t) = (
						(
							(x(i, j) - grid(j, t)) / (grid(j, t + k) - grid(j, t)) *
							(*prev_bases)(i, j, t)
						) +
						(
							(grid(j, t + k + 1) - x(i, j)) / (grid(j, t + k + 1) - grid(j, t + 1)) *
							(*prev_bases)(i, j, t + 1)
						)
					);
	}
}

void b_splines_derivative(Tensor& grid, Tensor& bases_minus_1, Tensor& coeff, uint32_t spline_order, Tensor& grad) {
	uint32_t num_samples = bases_minus_1.shape[0];
	uint32_t in_features = bases_minus_1.shape[1];
	uint32_t out_features = coeff.shape[0];
	uint32_t num_grid_points = grid.shape[1];
	uint32_t num_bases = num_grid_points - spline_order - 1;

	for (uint32_t i = 0; i < num_samples; ++i)
		for (uint32_t j = 0; j < in_features; ++j) {
			grad(i, j) = 0;
			for (uint32_t k = 0; k < out_features; ++k)
				for (uint32_t t = 0; t < num_bases; ++t)
					grad(i, j) += coeff(k, j, t) * (
						bases_minus_1(i, j, t) / (grid(j, t + spline_order) - grid(j, t)) -
						bases_minus_1(i, j, t + 1) / (grid(j, t + spline_order + 1) - grid(j, t + 1))
					);
			grad(i, j) *= spline_order;
		}
}

Tensor b_splines(Tensor& grid, Tensor& x, uint32_t spline_order, Tensor& coeff) {
	if (spline_order <= 1) {
		std::cerr << "spline_order must larger than 1." << std::endl;
		throw;
	}

	uint32_t num_samples = x.shape[0];
	uint32_t in_features = x.shape[1];
	uint32_t num_grid_points = grid.shape[1];
	uint32_t num_bases = num_grid_points - spline_order - 1;

	float *bases_temp_data = new float[num_samples * in_features * (num_grid_points - 1)];
	Tensor bases_temp({num_samples, in_features, num_grid_points - 1}, bases_temp_data);

	float *bases_minus_1_data = new float[num_samples * in_features * (num_grid_points - spline_order)];
	Tensor bases_minus_1({num_samples, in_features, num_grid_points - spline_order}, bases_minus_1_data);

	float *bases_data = new float[num_samples * in_features * (num_grid_points - spline_order - 1)];
	Tensor bases({num_samples, in_features, num_grid_points - spline_order - 1}, bases_data);

	b_splines(grid, x, spline_order, &bases, &bases_temp, &bases_minus_1);

	// return bases;

	float *grad_data = new float[num_samples * in_features];
	Tensor grad({num_samples, in_features}, grad_data);

	b_splines_derivative(grid, bases_minus_1, coeff, spline_order, grad);

	return grad;
}

}
#endif