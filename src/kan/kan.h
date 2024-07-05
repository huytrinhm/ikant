#ifndef KAN_H
#define KAN_H

#include <iostream>
#include <vector>
#include "tensor.h"
#include "spline.h"

namespace KAN {
constexpr float GRID_MIN = -1;
constexpr float GRID_MAX = 1;

struct KANLayer {
	// metadata
	uint32_t in_features;
	uint32_t out_features;

	// params
	Tensor coeff;
	Tensor basis_weights;
	Tensor spline_weights;
	Tensor biases;

	// params grad
	Tensor coeff_grad;
	Tensor basis_weights_grad;
	Tensor spline_weights_grad;
	Tensor biases_grad;

	// saved data from forward pass
	Tensor bases;
	Tensor bases_minus_1;

	// activations
	Tensor activations;
};

struct KANNet {
	// metadata
	uint32_t num_layers;
	uint32_t spline_order;

	// pre-allocated memory
	float *params_data;
	float *params_grad_data;
	float *activations_data;

	// spline grid
	Tensor grid;

	// pre-allocated temp memory
	Tensor bases_temp;
	Tensor partial_grad;

	// layers
	KANLayer *layers;
};

void grid_init(Tensor &grid, uint32_t spline_order, uint32_t grid_size) {
	uint32_t num_grid_points = grid_size + 2 * spline_order + 1;
	grid = Tensor({num_grid_points}, new float[num_grid_points]);

	float h = (GRID_MAX - GRID_MIN) / grid_size;
	grid(0) = GRID_MIN - spline_order * h;
	for (uint32_t i = 1; i < num_grid_points; ++i)
		grid(i) = grid(i - 1) + h;
}

void KANLayer_init(KANLayer &layer, uint32_t in_features, uint32_t out_features, uint32_t num_bases, float* &params, float* &params_grad, float* &activations) {
	layer.in_features = in_features;
	layer.out_features = out_features;

	layer.coeff = Tensor({out_features, in_features, num_bases}, params);
	layer.coeff_grad = Tensor({out_features, in_features, num_bases}, params_grad);
	params += out_features * in_features * num_bases;
	params_grad += out_features * in_features * num_bases;

	layer.basis_weights = Tensor({out_features}, params);
	layer.basis_weights_grad = Tensor({out_features}, params_grad);
	params += out_features;
	params_grad += out_features;

	layer.spline_weights = Tensor({out_features}, params);
	layer.spline_weights_grad = Tensor({out_features}, params_grad);
	params += out_features;
	params_grad += out_features;

	layer.biases = Tensor({out_features}, params);
	layer.biases_grad = Tensor({out_features}, params_grad);
	params += out_features;
	params_grad += out_features;

	layer.bases = Tensor({in_features, num_bases}, new float[in_features * num_bases]);
	layer.bases_minus_1 = Tensor({in_features, num_bases - 1}, new float[in_features * (num_bases - 1)]);

	layer.activations = Tensor({out_features}, activations);
	activations += out_features;
}

KANNet KANNet_create(std::vector<uint32_t> widths, uint32_t spline_order, uint32_t grid_size) {
	if (widths.size() < 2) {
		std::cerr << "KANNet must have at least 1 layer (widths.size() >= 2)." << std::endl;
		throw;
	}

	KANNet net;
	net.num_layers = widths.size() - 1;
	net.spline_order = spline_order;
	uint32_t num_bases = grid_size + spline_order;
	uint32_t params_size = 0;
	uint32_t activations_size = 0;
	uint32_t max_width = widths[0];

	for (uint32_t l = 0; l < net.num_layers; l++) {
		uint32_t in_features = widths[l];
		uint32_t out_features = widths[l + 1];

		params_size += (
			// coeff
			out_features * in_features * num_bases +
			// basis_weights
			out_features +
			// spline weights
			out_features +
			// biases
			out_features
		);

		activations_size += out_features;

		max_width = max(max_width, in_features);
	}

	net.params_data = new float[params_size];
	net.params_grad_data = new float[params_size];
	net.activations_data = new float[activations_size];

	grid_init(net.grid, spline_order, grid_size);
	net.bases_temp = Tensor({max_width, num_bases + spline_order}, new float[max_width * (num_bases + spline_order)]);
	net.partial_grad = Tensor({max_width}, new float[max_width]);

	net.layers = new KANLayer[net.num_layers];

	float *params = net.params_data;
	float *params_grad = net.params_grad_data;
	float *activations = net.activations_data;

	for (uint32_t l = 0; l < net.num_layers; l++) {
		KANLayer_init(
			net.layers[l],
			widths[l], widths[l + 1],
			num_bases,
			params,
			params_grad,
			activations
		);
	}

	return net;
}

}
#endif