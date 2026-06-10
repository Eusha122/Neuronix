#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "neuronix/layers/layer.hpp"
#include "neuronix/layers/conv2d.hpp"

namespace neuronix {

// Residual block with identity shortcut.
//
//   x ──────────────────────────────(+)── ReLU ── out
//        └─ Conv ── ReLU ── Conv ──┘
//
// Both convolutions keep spatial dimensions unchanged (same-padding).
// C_in must equal C_out so the identity shortcut can be added directly.
// Use kernel_size=3, padding=1 (the default) for the standard ResNet block.
class ResidualBlock : public Layer {
public:
    ResidualBlock(std::size_t channels,
                  std::size_t height,
                  std::size_t width,
                  std::size_t kernel_size = 3,
                  std::size_t padding     = 1);

    // Inference — no caching
    [[nodiscard]] Matrix forward(const Matrix& input) const override;

    // Training forward — caches pre-ReLU activations for backward
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;

    Matrix backward(const Matrix& grad_output) override;

    void update(double lr) override;
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;
    void adamw_step(double lr, double beta1, double beta2,
                    double eps, double wd, std::size_t t) override;
    void zero_grad() override;

    [[nodiscard]] std::string_view name()        const noexcept override { return "ResidualBlock"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override;

    // Accessors for serialization
    [[nodiscard]] std::size_t channels()    const noexcept { return C_; }
    [[nodiscard]] std::size_t height()      const noexcept { return H_; }
    [[nodiscard]] std::size_t width()       const noexcept { return W_; }
    [[nodiscard]] std::size_t kernel_size() const noexcept { return K_; }
    [[nodiscard]] std::size_t padding()     const noexcept { return pad_; }
    [[nodiscard]] const Conv2D& conv1()     const noexcept { return *conv1_; }
    [[nodiscard]] const Conv2D& conv2()     const noexcept { return *conv2_; }

private:
    std::size_t C_, H_, W_, K_, pad_;

    std::unique_ptr<Conv2D> conv1_;
    std::unique_ptr<Conv2D> conv2_;

    Matrix cache_input_;     // input x, needed for skip gradient
    Matrix cache_conv1_out_; // conv1 output before ReLU, needed for relu1 backward
    Matrix cache_sum_;       // conv2_out + x before final ReLU, for relu2 backward
};

} // namespace neuronix
