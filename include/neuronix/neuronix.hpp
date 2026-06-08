#pragma once

#include <string_view>

#include "neuronix/version.hpp"
#include "neuronix/core/matrix.hpp"
#include "neuronix/core/tensor.hpp"
#include "neuronix/layers/layer.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/layers/conv2d.hpp"
#include "neuronix/layers/max_pool2d.hpp"
#include "neuronix/layers/batch_norm.hpp"
#include "neuronix/layers/flatten.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/tanh_activation.hpp"
#include "neuronix/activations/softmax.hpp"
#include "neuronix/activations/dropout.hpp"
#include "neuronix/losses/loss.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/losses/cross_entropy_loss.hpp"
#include "neuronix/models/model.hpp"
#include "neuronix/optimizers/sgd.hpp"
#include "neuronix/optimizers/adam.hpp"
#include "neuronix/optimizers/lr_scheduler.hpp"
#include "neuronix/data/mnist_loader.hpp"
#include "neuronix/data/data_loader.hpp"
#include "neuronix/metrics/accuracy.hpp"

namespace neuronix {

[[nodiscard]] std::string_view framework_name() noexcept;

} // namespace neuronix
