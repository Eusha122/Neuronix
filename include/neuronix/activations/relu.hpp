#pragma once

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class ReLU : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return x > 0.0 ? x : 0.0; });
    }
    [[nodiscard]] std::string_view name() const noexcept override { return "ReLU"; }
};

} // namespace neuronix
