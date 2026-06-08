#pragma once

#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// Flatten layer — no-op in Neuronix's (features, N) column layout.
//
// In this framework every layer already stores one flattened sample per column,
// so Conv2D → Dense needs no data reshaping. Flatten is inserted purely to make
// the transition visible in the model summary and to match the conventional
// mental model of a ConvNet graph.
//
// forward / backward both pass data through unchanged.
class Flatten : public Layer {
public:
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        return input;
    }

    Matrix backward(const Matrix& grad_output) override {
        return grad_output;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "Flatten"; }
    [[nodiscard]] std::size_t param_count() const noexcept override { return 0; }
};

} // namespace neuronix
