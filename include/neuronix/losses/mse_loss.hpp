#pragma once

#include "neuronix/losses/loss.hpp"

namespace neuronix {

class MSELoss : public Loss {
public:
    // predicted and target must have the same shape.
    // loss = mean((predicted - target)^2)
    // grad = 2 * (predicted - target) / total_elements
    double forward(const Matrix& predicted, const Matrix& target) override;
    [[nodiscard]] Matrix backward() const override { return grad_; }

private:
    Matrix grad_;
};

} // namespace neuronix
