#pragma once

#include <cmath>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Tanh : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return std::tanh(x); });
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "Tanh"; }
};

} // namespace neuronix
