#pragma once

#include <cmath>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Sigmoid : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "Sigmoid"; }
};

} // namespace neuronix
