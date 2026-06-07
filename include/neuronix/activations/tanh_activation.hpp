#pragma once

#include <cmath>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Tanh : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return std::tanh(x); });
    }

    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        output_cache_ = forward(input);
        return output_cache_;
    }

    Matrix backward(const Matrix& grad_output) override {
        // d(tanh)/dx = 1 - tanh(x)^2
        Matrix deriv = output_cache_.apply([](double t) { return 1.0 - t * t; });
        return grad_output.hadamard(deriv);
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "Tanh"; }

private:
    Matrix output_cache_;
};

} // namespace neuronix
