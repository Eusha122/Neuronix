#pragma once

#include <string_view>

#include "neuronix/version.hpp"
#include "neuronix/core/matrix.hpp"
#include "neuronix/core/tensor.hpp"

namespace neuronix {

[[nodiscard]] std::string_view framework_name() noexcept;

} // namespace neuronix
