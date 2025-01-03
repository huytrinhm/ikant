#include "kan.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include "lapacke.h"
#include "spline.h"
#include "tensor.h"

namespace KAN {
constexpr float GRID_MIN = -1;
constexpr float GRID_MAX = 1;

std::ostream& operator<<(std::ostream& os, const KANLayer& layer) {
  os << "KANLayer:\n";
  os << "\tin_features: " << layer.in_features << "\n";
  os << "\tout_features: " << layer.out_features << "\n";
  os << "\tcoeff: " << layer.coeff.dim << " (" << layer.coeff.shape[0] << ' '
     << layer.coeff.shape[1] << ' ' << layer.coeff.shape[2] << ") at "
     << layer.coeff.data << "\n";
  os << "\tcoeff_grad: " << layer.coeff_grad.dim << " ("
     << layer.coeff_grad.shape[0] << ' ' << layer.coeff_grad.shape[1] << ' '
     << layer.coeff_grad.shape[2] << ") at " << layer.coeff_grad.data << "\n";
  os << "\tbases: " << layer.bases.dim << " (" << layer.bases.shape[0] << ' '
     << layer.bases.shape[1] << ") at " << layer.bases.data << "\n";
  os << "\tbases_minus_1: " << layer.bases_minus_1.dim << " ("
     << layer.bases_minus_1.shape[0] << ' ' << layer.bases_minus_1.shape[1]
     << ") at " << layer.bases_minus_1.data << "\n";
  os << "\tsplines: " << layer.splines.dim << " (" << layer.splines.shape[0]
     << ' ' << layer.splines.shape[1] << ") at " << layer.splines.data << "\n";
  os << "\tresiduals: " << layer.residuals.dim << " ("
     << layer.residuals.shape[0] << ") at " << layer.residuals.data << "\n";
  os << "\tactivations: " << layer.activations.dim << " ("
     << layer.activations.shape[0] << ") at " << layer.activations.data << "\n";
  os << "\tpartial_grad: " << layer.partial_grad.dim << " ("
     << layer.partial_grad.shape[0] << ") at " << layer.partial_grad.data
     << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const KANNet& net) {
  os << "KANNet:\n";
  os << "\tnum_layers: " << net.num_layers << "\n";
  os << "\tspline_order: " << net.spline_order << "\n";
  os << "\tparams_data: " << net.params_data << "\n";
  os << "\tparams_grad_data: " << net.params_grad_data << "\n";
  os << "\tactivations_data: " << net.activations_data << "\n";
  os << "\tgrid: " << net.grid << "\n";
  os << "\tbases_temp: " << net.bases_temp.dim << " ("
     << net.bases_temp.shape[0] << ' ' << net.bases_temp.shape[1] << ") at "
     << net.bases_temp.data << "\n";

  os << "===LAYERS===\n";
  for (uint32_t l = 0; l < net.num_layers; ++l)
    os << net.layers[l] << "\n";

  return os;
}

void grid_init(Tensor& grid, uint32_t spline_order, uint32_t grid_size) {
  uint32_t num_grid_points = grid_size + 2 * spline_order + 1;
  grid = Tensor({num_grid_points});

  float h = (GRID_MAX - GRID_MIN) / grid_size;
  grid(0) = GRID_MIN - spline_order * h;
  for (uint32_t i = 1; i < num_grid_points; ++i)
    grid(i) = grid(i - 1) + h;
}

void KANLayer_init(KANLayer& layer,
                   uint32_t in_features,
                   uint32_t out_features,
                   uint32_t num_bases,
                   float*& params,
                   float*& params_grad,
                   float*& activations) {
  layer.in_features = in_features;
  layer.out_features = out_features;

  layer.coeff = Tensor({out_features, in_features, num_bases}, params);
  layer.coeff_grad =
      Tensor({out_features, in_features, num_bases}, params_grad);
  params += out_features * in_features * num_bases;
  params_grad += out_features * in_features * num_bases;

  layer.basis_weights = Tensor({out_features, in_features}, params);
  layer.basis_weights_grad = Tensor({out_features, in_features}, params_grad);
  params += out_features * in_features;
  params_grad += out_features * in_features;

  layer.spline_weights = Tensor({out_features, in_features}, params);
  layer.spline_weights_grad = Tensor({out_features, in_features}, params_grad);
  params += out_features * in_features;
  params_grad += out_features * in_features;

  layer.biases = Tensor({out_features}, params);
  layer.biases_grad = Tensor({out_features}, params_grad);
  params += out_features;
  params_grad += out_features;

  layer.bases = Tensor({in_features, num_bases});
  layer.bases_minus_1 = Tensor({in_features, num_bases + 1});
  layer.splines = Tensor({out_features, in_features});
  layer.edge_activations = Tensor({out_features, in_features});
  layer.residuals = Tensor({in_features});
  layer.partial_grad = Tensor({out_features});

  layer.activations = Tensor({out_features}, activations);
  activations += out_features;

  layer.mask = Tensor({out_features});
  for (uint32_t i = 0; i < out_features; i++)
    layer.mask(i) = 1;
}

void KANNet_load_checkpoint(const char* filename,
                            uint32_t& num_layers,
                            uint32_t& spline_order,
                            uint32_t& grid_size,
                            uint32_t*& widths,
                            float*& params_data) {
  std::ifstream fin(filename, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw;
  }

  fin.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(&spline_order), sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(&grid_size), sizeof(uint32_t));

  widths = new uint32_t[num_layers + 1];
  fin.read(reinterpret_cast<char*>(widths),
           sizeof(uint32_t) * (num_layers + 1));

  uint32_t num_bases = grid_size + spline_order;
  uint32_t params_size = 0;

  for (uint32_t l = 0; l < num_layers; l++) {
    uint32_t in_features = widths[l];
    uint32_t out_features = widths[l + 1];

    params_size += (
        // coeff
        out_features * in_features * num_bases +
        // basis_weights
        out_features * in_features +
        // spline weights
        out_features * in_features +
        // biases
        out_features);
  }

  params_data = new float[params_size];
  fin.read(reinterpret_cast<char*>(params_data), sizeof(float) * params_size);

  fin.close();
}

void KANNet_save_checkpoint(KANNet& net, const char* filename) {
  std::ofstream fout(filename, std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw;
  }

  uint32_t grid_size = net.grid_size;

  fout.write(reinterpret_cast<char*>(&net.num_layers), sizeof(uint32_t));
  fout.write(reinterpret_cast<char*>(&net.spline_order), sizeof(uint32_t));
  fout.write(reinterpret_cast<char*>(&grid_size), sizeof(uint32_t));

  uint32_t num_bases = grid_size + net.spline_order;

  for (uint32_t l = 0; l < net.num_layers; l++) {
    fout.write(reinterpret_cast<char*>(&net.layers[l].in_features),
               sizeof(uint32_t));
  }

  fout.write(
      reinterpret_cast<char*>(&net.layers[net.num_layers - 1].out_features),
      sizeof(uint32_t));
  fout.write(reinterpret_cast<char*>(net.params_data),
             sizeof(float) * net.num_params);

  fout.close();
}

float random_uniform(float limit) {
  return (static_cast<float>(std::rand()) / RAND_MAX) * 2 * limit - limit;
}

float invSqrt(float number) {
  union {
    float f;
    uint32_t i;
  } conv;

  float x2;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  conv.f = number;
  conv.i = 0x5f3759df - (conv.i >> 1);
  conv.f = conv.f * (threehalfs - (x2 * conv.f * conv.f));
  return conv.f;
}

void KANNet_weight_init(KANNet& net) {
  uint32_t num_bases = net.grid_size + net.spline_order;
  float spline_noise = 0.5 * 0.1 / net.grid_size;

  Tensor bases({num_bases, num_bases});
  Tensor bases_temp({num_bases, net.grid_size + 2 * net.spline_order});

  Tensor x({num_bases});
  float h = (GRID_MAX - GRID_MIN) / (num_bases - 1);
  x(0) = GRID_MIN;
  for (uint32_t i = 1; i < num_bases; ++i)
    x(i) = x(i - 1) + h;

  for (uint32_t l = 0; l < net.num_layers; l++) {
    KANLayer& layer = net.layers[l];

    b_splines(net.grid, x, net.spline_order, &bases, &bases_temp, nullptr);

    for (uint32_t i = 0; i < layer.in_features; i++) {
      float inv_sqrt_in_features = invSqrt(layer.in_features);
      for (uint32_t j = 0; j < layer.out_features; j++) {
        layer.basis_weights(j, i) = inv_sqrt_in_features + random_uniform(0.1);
        layer.spline_weights(j, i) = 1;

        for (uint32_t k = 0; k < num_bases; k++) {
          layer.coeff(j, i, k) = random_uniform(spline_noise);
        }
      }
    }

    // std::cout << layer.basis_weights << std::endl;
    // std::cout << layer.spline_weights << std::endl;
    // std::cout << bases << std::endl;
    // std::cout << layer.coeff << std::endl;

    LAPACKE_sgels(LAPACK_COL_MAJOR, 'T', num_bases, num_bases,
                  layer.in_features * layer.out_features, bases.data, num_bases,
                  layer.coeff.data, num_bases);

    // std::cout << "solution\n";
    // std::cout << layer.coeff << std::endl;

    for (uint32_t i = 0; i < layer.out_features; i++)
      layer.biases(i) = 0;
  }
}

KANNet KANNet_create(std::vector<uint32_t> widths,
                     uint32_t spline_order,
                     uint32_t grid_size,
                     float* params_data) {
  if (widths.size() < 2) {
    std::cerr << "KANNet must have at least 1 layer (widths.size() >= 2)."
              << std::endl;
    throw;
  }

  KANNet net;
  net.num_layers = widths.size() - 1;
  net.spline_order = spline_order;
  net.grid_size = grid_size;
  uint32_t num_bases = grid_size + spline_order;
  uint32_t activations_size = 0;
  uint32_t max_width = widths[0];

  net.num_params = 0;
  for (uint32_t l = 0; l < net.num_layers; l++) {
    uint32_t in_features = widths[l];
    uint32_t out_features = widths[l + 1];

    net.num_params += (
        // coeff
        out_features * in_features * num_bases +
        // basis_weights
        out_features * in_features +
        // spline weights
        out_features * in_features +
        // biases
        out_features);

    activations_size += out_features;

    max_width = std::max(max_width, in_features);
  }

  if (params_data)
    net.params_data = params_data;
  else
    net.params_data = new float[net.num_params];
  net.params_grad_data = new float[net.num_params];
  net.activations_data = new float[activations_size];

  grid_init(net.grid, spline_order, grid_size);
  net.bases_temp = Tensor({max_width, num_bases + spline_order});

  net.layers = new KANLayer[net.num_layers];

  float* params = net.params_data;
  float* params_grad = net.params_grad_data;
  float* activations = net.activations_data;

  for (uint32_t l = 0; l < net.num_layers; l++) {
    KANLayer_init(net.layers[l], widths[l], widths[l + 1], num_bases, params,
                  params_grad, activations);
  }

  if (!params_data)
    KANNet_weight_init(net);

  return net;
}

float SiLU(float x) {
  if (x > 0)
    return x / (1 + std::expf(-x));
  else {
    float t = std::expf(x);
    return x * t / (1 + t);
  }
}

float SiLU_derivative(float x) {
  float sigmoid;

  if (x > 0)
    sigmoid = 1 / (1 + std::expf(-x));
  else {
    float t = std::expf(x);
    sigmoid = t / (1 + t);
  }

  return sigmoid * (1 + x * (1 - sigmoid));
}

void KANLayer_neuron_forward(KANLayer& layer,
                             uint32_t spline_order,
                             uint32_t i) {
  uint32_t num_bases = layer.bases.shape[1];

  layer.activations(i) = 0;

  for (uint32_t j = 0; j < layer.in_features; ++j) {
    layer.splines(i, j) = 0;
    for (uint32_t b = 0; b < num_bases; ++b) {
      layer.splines(i, j) += layer.coeff(i, j, b) * layer.bases(j, b);
    }

    float edge_act = layer.spline_weights(i, j) * layer.splines(i, j) +
                     layer.basis_weights(i, j) * layer.residuals(j);
    layer.edge_activations(i, j) += layer.mask(i) * std::fabs(edge_act);
    layer.activations(i) += edge_act;
  }

  layer.activations(i) += layer.biases(i);
}

void KANLayer_forward(KANLayer& layer,
                      Tensor& inputs,
                      Tensor& grid,
                      uint32_t spline_order,
                      Tensor& bases_temp) {
  b_splines(grid, inputs, spline_order, &layer.bases, &bases_temp,
            &layer.bases_minus_1);
  for (uint32_t i = 0; i < layer.in_features; ++i) {
    layer.residuals(i) = SiLU(inputs(i));
  }

  for (uint32_t i = 0; i < layer.out_features; ++i) {
    KANLayer_neuron_forward(layer, spline_order, i);
    layer.activations(i) *= layer.mask(i);
  }
}

void KANNet_forward(KANNet& net, Tensor& x) {
  Tensor* inputs = &x;
  for (uint32_t l = 0; l < net.num_layers; ++l) {
    KANLayer_forward(net.layers[l], *inputs, net.grid, net.spline_order,
                     net.bases_temp);
    inputs = &net.layers[l].activations;
  }
}

void KANLayer_backward(KANLayer& layer,
                       Tensor& inputs,
                       Tensor* next_grad,
                       Tensor& grid,
                       uint32_t spline_order,
                       float lambda) {
  for (uint32_t i = 0; i < layer.out_features; ++i) {
    layer.biases_grad(i) += layer.mask(i) * (layer.partial_grad(i) - lambda);

    for (uint32_t j = 0; j < layer.in_features; ++j) {
      layer.basis_weights_grad(i, j) +=
          layer.mask(i) * layer.partial_grad(i) * layer.residuals(j);

      layer.spline_weights_grad(i, j) +=
          layer.mask(i) * layer.partial_grad(i) * layer.splines(i, j);

      for (uint32_t b = 0; b < layer.bases.shape[1]; ++b) {
        layer.coeff_grad(i, j, b) += layer.mask(i) * layer.partial_grad(i) *
                                     layer.spline_weights(i, j) *
                                     layer.bases(j, b);
      }
    }
  }

  if (!next_grad)
    return;

  b_splines_derivative(grid, layer.bases_minus_1, layer.coeff,
                       layer.partial_grad, layer.spline_weights, spline_order,
                       *next_grad);

  for (uint32_t j = 0; j < layer.in_features; ++j) {
    for (uint32_t i = 0; i < layer.out_features; ++i) {
      (*next_grad)(j) +=
          layer.mask(i) * (layer.partial_grad(i) * layer.basis_weights(i, j) *
                               SiLU_derivative(inputs(j)) +
                           lambda);
    }
  }
}

void KANNet_backward(KANNet& net, Tensor& x, Tensor& y, float lambda) {
  for (uint32_t i = 0; i < net.layers[net.num_layers - 1].out_features; ++i) {
    net.layers[net.num_layers - 1].partial_grad(i) =
        2 * (net.layers[net.num_layers - 1].activations(i) - y(i)) + lambda;
  }

  for (uint32_t l = net.num_layers - 1; l > 0; --l) {
    KANLayer_backward(net.layers[l], net.layers[l - 1].activations,
                      &net.layers[l - 1].partial_grad, net.grid,
                      net.spline_order, lambda);
  }

  KANLayer_backward(net.layers[0], x, nullptr, net.grid, net.spline_order,
                    lambda);
}

void KANNet_zero_grad(KANNet& net) {
  for (uint32_t i = 0; i < net.num_params; i++)
    net.params_grad_data[i] = 0;
}

float MSELoss(Tensor& output, Tensor& target) {
  float accum = 0;
  for (uint32_t i = 0; i < output.shape[0]; ++i) {
    float diff = output(i) - target(i);
    accum = diff * diff;
  }

  return accum;
}

void KANNet_get_spline_range(KANNet& net,
                             Tensor& X,
                             std::vector<std::vector<float>>& min_act,
                             std::vector<std::vector<float>>& max_act,
                             bool reset) {
  for (uint32_t i = 0; i < X.shape[0]; ++i) {
    min_act[0][i] = reset ? X(i) : std::min(min_act[0][i], X(i));
    max_act[0][i] = reset ? X(i) : std::max(max_act[0][i], X(i));
  }

  for (uint32_t l = 0; l < net.num_layers; ++l) {
    for (uint32_t i = 0; i < net.layers[l].out_features; ++i) {
      min_act[l + 1][i] =
          reset ? net.layers[l].activations(i)
                : std::min(min_act[l + 1][i], net.layers[l].activations(i));
      max_act[l + 1][i] =
          reset ? net.layers[l].activations(i)
                : std::max(max_act[l + 1][i], net.layers[l].activations(i));
    }
  }
}

float KANNet_run_epoch(KANNet& net,
                       Tensor& X,
                       Tensor& y,
                       float lr,
                       float lambda) {
  KANNet_zero_grad(net);
  float epoch_loss = 0;

  for (uint32_t i = 0; i < X.shape[0]; ++i) {
    Tensor sample(1, &X.shape[1], &X.stride[1], &X(i, 0));
    Tensor gt(1, &y.shape[1], &y.stride[1], &y(i, 0));
    KANNet_forward(net, sample);
    KANNet_backward(net, sample, gt, lambda);
    epoch_loss += MSELoss(net.layers[net.num_layers - 1].activations, gt);
  }

  for (uint32_t i = 0; i < net.num_params; i++) {
    net.params_data[i] -= lr * net.params_grad_data[i] / X.shape[0];
  }

  return epoch_loss / X.shape[0];
}

}  // namespace KAN