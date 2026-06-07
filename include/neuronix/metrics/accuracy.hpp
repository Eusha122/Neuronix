#pragma once

#include <vector>

#include "neuronix/core/matrix.hpp"

namespace neuronix::metrics {

// Returns the fraction of samples where argmax(predicted column) == true_label.
// predicted: (classes, N) — logits or probabilities
// true_labels: N raw class indices
[[nodiscard]] double accuracy(const Matrix& predicted,
                              const std::vector<int>& true_labels);

// Returns the fraction of samples where argmax(predicted) == argmax(one_hot).
// Both matrices must be (classes, N).
[[nodiscard]] double accuracy(const Matrix& predicted, const Matrix& one_hot);

} // namespace neuronix::metrics
