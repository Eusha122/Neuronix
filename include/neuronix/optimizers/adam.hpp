#pragma once

#include <cstddef>

#include "neuronix/models/model.hpp"

namespace neuronix {

class Adam {
public:
    explicit Adam(Model& model,
                  double lr    = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double eps   = 1e-8) noexcept
        : model_{model}, lr_{lr}, beta1_{beta1}, beta2_{beta2}, eps_{eps} {}

    void step() {
        ++t_;
        model_.adam_step(lr_, beta1_, beta2_, eps_, t_);
    }

    void zero_grad() { model_.zero_grad(); }

    [[nodiscard]] double      lr()         const noexcept { return lr_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return t_; }

    void set_lr(double lr) noexcept { lr_ = lr; }

private:
    Model&      model_;
    double      lr_;
    double      beta1_;
    double      beta2_;
    double      eps_;
    std::size_t t_{0};
};

} // namespace neuronix
