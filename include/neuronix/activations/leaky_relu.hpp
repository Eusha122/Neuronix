#pragma once

#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// LeakyReLU: f(x) = x if x > 0, else alpha * x
// Prevents the "dying ReLU" problem — negative inputs still receive gradient.
// Default alpha=0.01 (standard Leaky ReLU).
class LeakyReLU : public Layer {
public:
    explicit LeakyReLU(double alpha = 0.01) noexcept : alpha_{alpha} {}

    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        const double a = alpha_;
        return input.apply([a](double x) { return x > 0.0 ? x : a * x; });
    }

    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        cache_ = input;
        return forward(input);
    }

    Matrix backward(const Matrix& grad_output) override {
        const double a = alpha_;
        return grad_output.hadamard(
            cache_.apply([a](double x) { return x > 0.0 ? 1.0 : a; }));
    }

    [[nodiscard]] std::string_view name()        const noexcept override { return "LeakyReLU"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override { return 0; }
    [[nodiscard]] double           alpha()        const noexcept          { return alpha_; }

private:
    double alpha_;
    Matrix cache_;
};

} // namespace neuronix
