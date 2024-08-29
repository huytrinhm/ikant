#ifndef KAN_H
#define KAN_H

#include <cstdint>
#include <iostream>
#include <vector>
#include "tensor.h"

namespace KAN {

struct KANLayer {
  // metadata
  uint32_t in_features;
  uint32_t out_features;

  // params
  Tensor coeff;           // out * in * bases
  Tensor basis_weights;   // out * in
  Tensor spline_weights;  // out * in
  Tensor biases;          // out

  // params grad
  Tensor coeff_grad;
  Tensor basis_weights_grad;
  Tensor spline_weights_grad;
  Tensor biases_grad;

  // saved data from forward pass
  Tensor bases;             // in * bases
  Tensor bases_minus_1;     // in * (bases - 1)
  Tensor splines;           // out * in
  Tensor residuals;         // in
  Tensor edge_activations;  // out * in

  Tensor activations;   // out
  Tensor partial_grad;  // out
};

struct KANNet {
  // metadata
  uint32_t num_layers;
  uint32_t spline_order;
  uint32_t num_params;
  uint32_t grid_size;

  // pre-allocated memory
  float* params_data;
  float* params_grad_data;
  float* activations_data;

  // spline grid
  Tensor grid;  // num_grid_points = grid_size + 2*spline_order + 1

  // pre-allocated temp memory
  Tensor bases_temp;

  // layers
  KANLayer* layers;
};

std::ostream& operator<<(std::ostream& os, const KANLayer& layer);

std::ostream& operator<<(std::ostream& os, const KANNet& net);

void grid_init(Tensor& grid, uint32_t spline_order, uint32_t grid_size);

void KANLayer_init(KANLayer& layer,
                   uint32_t in_features,
                   uint32_t out_features,
                   uint32_t num_bases,
                   float*& params,
                   float*& params_grad,
                   float*& activations);

void KANNet_load_checkpoint(const char* filename,
                            uint32_t& num_layers,
                            uint32_t& spline_order,
                            uint32_t& grid_size,
                            uint32_t*& widths,
                            float*& params_data);

void KANNet_save_checkpoint(KANNet& net, const char* filename);

void KANNet_weight_init(KANNet& net);

KANNet KANNet_create(std::vector<uint32_t> widths,
                     uint32_t spline_order,
                     uint32_t grid_size,
                     float* params_data = nullptr);

float SiLU(float x);

float SiLU_derivative(float x);

void KANLayer_neuron_forward(KANLayer& layer,
                             uint32_t spline_order,
                             uint32_t i);

void KANLayer_forward(KANLayer& layer,
                      Tensor& inputs,
                      Tensor& grid,
                      uint32_t spline_order,
                      Tensor& bases_temp);

void KANNet_forward(KANNet& net, Tensor& x);

void KANLayer_backward(KANLayer& layer,
                       Tensor& inputs,
                       Tensor* next_grad,
                       Tensor& grid,
                       uint32_t spline_order,
                       float lambda = 0.);

void KANNet_backward(KANNet& net, Tensor& x, Tensor& y, float lambda = 0.);

void KANNet_zero_grad(KANNet& net);

float MSELoss(Tensor& output, Tensor& target);

void KANNet_get_spline_range(KANNet& net,
                             Tensor& X,
                             std::vector<std::vector<float>>& min_act,
                             std::vector<std::vector<float>>& max_act,
                             bool reset = false);

float KANNet_run_epoch(KANNet& net,
                       Tensor& X,
                       Tensor& y,
                       float lr,
                       float lambda = 0.);

}  // namespace KAN
#endif