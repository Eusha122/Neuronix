#include "neuronix/layers/residual_block.hpp"

namespace neuronix {

ResidualBlock::ResidualBlock(std::size_t channels,
                               std::size_t height,
                               std::size_t width,
                               std::size_t kernel_size,
                               std::size_t padding)
    : C_{channels}, H_{height}, W_{width}, K_{kernel_size}, pad_{padding}
    , conv1_{std::make_unique<Conv2D>(channels, channels, kernel_size,
                                      height, width, padding)}
    , conv2_{std::make_unique<Conv2D>(channels, channels, kernel_size,
                                      height, width, padding)}
{}

// ── Inference (no caching) ────────────────────────────────────────────────────

Matrix ResidualBlock::forward(const Matrix& input) const {
    static const auto relu = [](double v){ return v > 0.0 ? v : 0.0; };

    Matrix h = conv1_->forward(input).apply(relu); // Conv → ReLU
    h = conv2_->forward(h) + input;                // Conv → skip add
    return h.apply(relu);                          // final ReLU
}

// ── Training forward (caches for backward) ───────────────────────────────────

Matrix ResidualBlock::forward_train(const Matrix& input) {
    static const auto relu = [](double v){ return v > 0.0 ? v : 0.0; };

    cache_input_    = input;
    cache_conv1_out_ = conv1_->forward_train(input);  // cache im2col inside conv1
    Matrix h         = cache_conv1_out_.apply(relu);  // ReLU1
    h                = conv2_->forward_train(h);      // cache im2col inside conv2
    cache_sum_       = h + input;                     // pre-ReLU2 sum (skip add)
    return cache_sum_.apply(relu);                    // ReLU2
}

// ── Backward ─────────────────────────────────────────────────────────────────

Matrix ResidualBlock::backward(const Matrix& grad_output) {
    // Backward through ReLU2: mask = (cache_sum_ > 0)
    Matrix grad = grad_output.hadamard(
        cache_sum_.apply([](double v){ return v > 0.0 ? 1.0 : 0.0; }));

    // At the addition: gradient flows equally to both paths
    Matrix grad_skip = grad;

    // Backward through Conv2
    grad = conv2_->backward(grad);

    // Backward through ReLU1: mask = (cache_conv1_out_ > 0)
    grad = grad.hadamard(
        cache_conv1_out_.apply([](double v){ return v > 0.0 ? 1.0 : 0.0; }));

    // Backward through Conv1
    grad = conv1_->backward(grad);

    // Add skip-path gradient (identity shortcut has gradient = 1)
    return grad + grad_skip;
}

// ── Parameter updates ─────────────────────────────────────────────────────────

void ResidualBlock::update(double lr) {
    conv1_->update(lr);
    conv2_->update(lr);
}

void ResidualBlock::adam_step(double lr, double beta1, double beta2,
                               double eps, std::size_t t) {
    conv1_->adam_step(lr, beta1, beta2, eps, t);
    conv2_->adam_step(lr, beta1, beta2, eps, t);
}

void ResidualBlock::adamw_step(double lr, double beta1, double beta2,
                                double eps, double wd, std::size_t t) {
    conv1_->adamw_step(lr, beta1, beta2, eps, wd, t);
    conv2_->adamw_step(lr, beta1, beta2, eps, wd, t);
}

void ResidualBlock::zero_grad() {
    conv1_->zero_grad();
    conv2_->zero_grad();
}

std::size_t ResidualBlock::param_count() const noexcept {
    return conv1_->param_count() + conv2_->param_count();
}

} // namespace neuronix
