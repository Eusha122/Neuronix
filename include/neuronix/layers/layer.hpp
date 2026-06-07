#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include "neuronix/core/matrix.hpp"

namespace neuronix {

class Layer {
public:
    virtual ~Layer() = default;

    // Inference — const, no side effects
    [[nodiscard]] virtual Matrix forward(const Matrix& input) const = 0;

    // Training forward — caches intermediates for backward. Default: calls forward().
    [[nodiscard]] virtual Matrix forward_train(const Matrix& input) {
        return forward(input);
    }

    // Backward — receives dL/d(output), returns dL/d(input), accumulates param grads.
    // Must call forward_train before backward.
    // Default: throws (layers without backward support use inference-only).
    virtual Matrix backward(const Matrix& /*grad_output*/) {
        throw std::logic_error{
            std::string{name()} + " does not support backward pass"};
    }

    // Apply gradient update with given learning rate. Default: no-op.
    virtual void update(double /*lr*/) {}

    // Zero accumulated gradients. Default: no-op.
    virtual void zero_grad() {}

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::size_t param_count() const noexcept { return 0; }
};

} // namespace neuronix
