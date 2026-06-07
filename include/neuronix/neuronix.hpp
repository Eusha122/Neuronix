#pragma once

#include <string_view>

#include "neuronix/version.hpp"
#include "neuronix/core/matrix.hpp"
#include "neuronix/core/tensor.hpp"
#include "neuronix/layers/layer.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/tanh_activation.hpp"
#include "neuronix/activations/softmax.hpp"
#include "neuronix/models/model.hpp"

namespace neuronix {

[[nodiscard]] std::string_view framework_name() noexcept;

} // namespace neuronix
