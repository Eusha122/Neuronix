#pragma once

#include "neuronix/losses/loss.hpp"

namespace neuronix {

// Numerically-stable Softmax + Cross-Entropy combined loss.
// predicted = logits (pre-softmax output of final Dense layer)
// target    = one-hot encoded labels, same shape as predicted
// loss      = -mean(sum_classes(target * log(softmax(logits))))
// grad      = (softmax(logits) - target) / batch_size
class CrossEntropyLoss : public Loss {
public:
    double forward(const Matrix& logits, const Matrix& target) override;
    [[nodiscard]] Matrix backward() const override { return grad_; }

private:
    Matrix grad_;
};

} // namespace neuronix
