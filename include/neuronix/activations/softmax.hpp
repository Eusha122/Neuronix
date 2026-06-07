#pragma once

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Softmax : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "Softmax"; }
};

} // namespace neuronix
