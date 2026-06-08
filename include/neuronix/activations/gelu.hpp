#pragma once

#include <cmath>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// GELU (Gaussian Error Linear Unit) — used in BERT, GPT, and most modern transformers.
// Approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Smooth alternative to ReLU — allows small negative values to pass through,
// weighted by the probability that the input is positive.
class GELU : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input.apply(fwd);
    }

    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        cache_ = input;
        return forward(input);
    }

    Matrix backward(const Matrix& grad_output) override {
        return grad_output.hadamard(cache_.apply(bwd));
    }

    [[nodiscard]] std::string_view name()        const noexcept override { return "GELU"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override { return 0; }

private:
    // sqrt(2/pi)
    static constexpr double kC = 0.7978845608028654;
    // coefficient in the cubic term
    static constexpr double kA = 0.044715;

    static double fwd(double x) noexcept {
        const double inner = kC * (x + kA * x * x * x);
        return 0.5 * x * (1.0 + std::tanh(inner));
    }

    static double bwd(double x) noexcept {
        const double inner = kC * (x + kA * x * x * x);
        const double t     = std::tanh(inner);
        const double sech2 = 1.0 - t * t;                        // sech²(inner)
        return 0.5 * (1.0 + t)
             + 0.5 * x * sech2 * kC * (1.0 + 3.0 * kA * x * x);
    }

    Matrix cache_;
};

} // namespace neuronix
