#pragma once

#include <cmath>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Sigmoid : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    }

    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        output_cache_ = forward(input);
        return output_cache_;
    }

    Matrix backward(const Matrix& grad_output) override {
        // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        Matrix deriv = output_cache_.hadamard(
            output_cache_.apply([](double s) { return 1.0 - s; }));
        return grad_output.hadamard(deriv);
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "Sigmoid"; }

private:
    Matrix output_cache_;
};

} // namespace neuronix
