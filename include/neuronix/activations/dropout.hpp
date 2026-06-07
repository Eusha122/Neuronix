#pragma once

#include <stdexcept>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Dropout : public Layer {
public:
    explicit Dropout(double drop_rate = 0.5) : drop_rate_{drop_rate} {
        if (drop_rate_ < 0.0 || drop_rate_ >= 1.0)
            throw std::invalid_argument{"Dropout: drop_rate must be in [0, 1)"};
    }

    // Inference: passthrough (expectation is preserved by inverted dropout scaling)
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input;
    }

    // Training: sample a Bernoulli mask, scale kept units by 1/(1-p)
    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        if (drop_rate_ == 0.0) return input;
        const double scale = 1.0 / (1.0 - drop_rate_);
        const double p     = drop_rate_;
        mask_ = Matrix::random_uniform(input.rows(), input.cols(), 0.0, 1.0)
                    .apply([p, scale](double u) { return u >= p ? scale : 0.0; });
        return input.hadamard(mask_);
    }

    // Backward: route gradient through the same kept units
    Matrix backward(const Matrix& grad_output) override {
        if (drop_rate_ == 0.0) return grad_output;
        return grad_output.hadamard(mask_);
    }

    [[nodiscard]] std::string_view name()      const noexcept override { return "Dropout"; }
    [[nodiscard]] double           drop_rate() const noexcept          { return drop_rate_; }

private:
    double drop_rate_;
    Matrix mask_;
};

} // namespace neuronix
