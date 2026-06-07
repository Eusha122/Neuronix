#pragma once

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class ReLU : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply([](double x) { return x > 0.0 ? x : 0.0; });
    }

    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        input_cache_ = input;
        return forward(input);
    }

    Matrix backward(const Matrix& grad_output) override {
        // dReLU/dx = 1 if x > 0, else 0
        Matrix mask = input_cache_.apply([](double x) { return x > 0.0 ? 1.0 : 0.0; });
        return grad_output.hadamard(mask);
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "ReLU"; }

private:
    Matrix input_cache_;
};

} // namespace neuronix
