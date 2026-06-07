#pragma once

#include <cstddef>
#include <string_view>

#include "neuronix/core/matrix.hpp"

namespace neuronix {

class Layer {
public:
    virtual ~Layer() = default;
    [[nodiscard]] virtual Matrix forward(const Matrix& input) const = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::size_t param_count() const noexcept { return 0; }
};

} // namespace neuronix
