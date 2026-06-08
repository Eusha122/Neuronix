#pragma once

#include <cstddef>

#include "neuronix/models/model.hpp"

namespace neuronix {

// AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
//
// Difference from Adam: weight decay is applied directly to the parameters
// BEFORE the gradient update, rather than being folded into the gradient.
// This gives better regularisation on adaptive-lr optimizers.
//
//   w *= (1 - lr * wd)          ← decoupled decay
//   m  = β₁ m + (1-β₁) g
//   v  = β₂ v + (1-β₂) g²
//   w -= lr * (m/bc₁) / (√(v/bc₂) + ε)
//
// Typical wd values: 0.01 (default), 0.1 for stronger regularisation.
class AdamW {
public:
    explicit AdamW(Model& model,
                   double lr    = 0.001,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double eps   = 1e-8,
                   double wd    = 0.01) noexcept
        : model_{model}, lr_{lr}, beta1_{beta1},
          beta2_{beta2}, eps_{eps}, wd_{wd} {}

    void step() {
        ++t_;
        model_.adamw_step(lr_, beta1_, beta2_, eps_, wd_, t_);
    }

    void zero_grad() { model_.zero_grad(); }

    [[nodiscard]] double      lr()         const noexcept { return lr_; }
    [[nodiscard]] double      wd()         const noexcept { return wd_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return t_; }

    void set_lr(double lr) noexcept { lr_ = lr; }
    void set_wd(double wd) noexcept { wd_ = wd; }

private:
    Model&      model_;
    double      lr_;
    double      beta1_;
    double      beta2_;
    double      eps_;
    double      wd_;
    std::size_t t_{0};
};

} // namespace neuronix
