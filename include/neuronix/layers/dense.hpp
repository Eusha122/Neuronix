#pragma once

#include <cstddef>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Dense : public Layer {
public:
    Dense(std::size_t in_features, std::size_t out_features);

    // Inference
    [[nodiscard]] Matrix forward(const Matrix& input) const override;

    // Training — caches input for backward
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;

    // Backward — accumulates grad_weights_ and grad_bias_, returns dL/dX
    Matrix backward(const Matrix& grad_output) override;

    // SGD update
    void update(double lr) override;

    // Adam update (maintains internal first/second moment state)
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;

    // AdamW: decoupled weight decay on weights_, then Adam update
    void adamw_step(double lr, double beta1, double beta2,
                    double eps, double wd, std::size_t t) override;

    // Zero accumulated gradients
    void zero_grad() override;

    [[nodiscard]] std::string_view name() const noexcept override { return "Dense"; }
    [[nodiscard]] std::size_t param_count() const noexcept override;

    [[nodiscard]] std::size_t in_features()  const noexcept { return in_; }
    [[nodiscard]] std::size_t out_features() const noexcept { return out_; }

    [[nodiscard]] const Matrix& weights()      const noexcept { return weights_; }
    [[nodiscard]] const Matrix& bias()         const noexcept { return bias_; }
    [[nodiscard]] const Matrix& grad_weights() const noexcept { return grad_weights_; }
    [[nodiscard]] const Matrix& grad_bias()    const noexcept { return grad_bias_; }

    void set_weights(Matrix w);
    void set_bias(Matrix b);

private:
    std::size_t in_;
    std::size_t out_;
    Matrix weights_;       // (out, in)
    Matrix bias_;          // (out, 1)
    Matrix grad_weights_;  // (out, in) accumulated gradient
    Matrix grad_bias_;     // (out, 1) accumulated gradient
    Matrix input_cache_;   // (in, batch) cached from forward_train
    // Adam moment matrices — zero-sized until first adam_step call
    Matrix m_w_, v_w_;     // first/second moment for weights (out, in)
    Matrix m_b_, v_b_;     // first/second moment for bias   (out, 1)
};

} // namespace neuronix
