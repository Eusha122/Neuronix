#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "neuronix/layers/layer.hpp"
#include "neuronix/losses/loss.hpp"

namespace neuronix {

class Model {
public:
    void add(std::unique_ptr<Layer> layer);

    template<typename L, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_unique<L>(std::forward<Args>(args)...));
    }

    // ── Inference ────────────────────────────────────────────────────────────
    [[nodiscard]] Matrix predict(const Matrix& input) const;

    // ── Training ─────────────────────────────────────────────────────────────

    // Forward pass that caches intermediates for backward.
    [[nodiscard]] Matrix train_forward(const Matrix& input);

    // Backward pass: propagates grad through layers in reverse.
    void backward(const Matrix& grad_output);

    // Apply one gradient-descent step to all trainable layers.
    void update(double lr);

    // Zero all accumulated gradients.
    void zero_grad();

    // Convenience: zero_grad → train_forward → loss.forward → backward → update.
    // Returns the loss value.
    double train_step(const Matrix& X, const Matrix& y, Loss& loss_fn, double lr);

    // Full training loop. Prints loss every print_every epochs when verbose=true.
    void fit(const Matrix& X, const Matrix& y, Loss& loss_fn,
             std::size_t epochs, double lr,
             bool verbose = false, std::size_t print_every = 100);

    // ── Metadata ──────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t depth() const noexcept { return layers_.size(); }
    [[nodiscard]] std::size_t total_params() const noexcept;
    void summary() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace neuronix
