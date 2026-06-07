#pragma once

#include "neuronix/models/model.hpp"

namespace neuronix {

class SGD {
public:
    SGD(Model& model, double lr) noexcept : model_{model}, lr_{lr} {}

    void step()      { model_.update(lr_); }
    void zero_grad() { model_.zero_grad(); }

    [[nodiscard]] double lr() const noexcept { return lr_; }
    void set_lr(double lr) noexcept { lr_ = lr; }

private:
    Model& model_;
    double lr_;
};

} // namespace neuronix
