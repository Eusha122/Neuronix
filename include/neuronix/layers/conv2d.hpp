#pragma once

#include <cstddef>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// 2-D convolution layer (stride = 1, configurable zero-padding).
//
// Input layout : (C_in  * H     * W,     N)  — each column is a flat (C,H,W) sample
// Output layout: (C_out * H_out * W_out, N)
//
// H_out = H + 2*padding - kernel_size + 1
// W_out = W + 2*padding - kernel_size + 1
class Conv2D : public Layer {
public:
    Conv2D(std::size_t in_channels,
           std::size_t out_channels,
           std::size_t kernel_size,
           std::size_t in_height,
           std::size_t in_width,
           std::size_t padding = 0);

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update(double lr) override;
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;
    void adamw_step(double lr, double beta1, double beta2,
                    double eps, double wd, std::size_t t) override;
    void zero_grad() override;

    [[nodiscard]] std::string_view name()        const noexcept override { return "Conv2D"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override;

    [[nodiscard]] std::size_t in_channels()  const noexcept { return in_; }
    [[nodiscard]] std::size_t out_channels() const noexcept { return out_; }
    [[nodiscard]] std::size_t kernel_size()  const noexcept { return K_; }
    [[nodiscard]] std::size_t in_height()    const noexcept { return H_; }
    [[nodiscard]] std::size_t in_width()     const noexcept { return W_; }
    [[nodiscard]] std::size_t out_height()   const noexcept { return H_out_; }
    [[nodiscard]] std::size_t out_width()    const noexcept { return W_out_; }
    [[nodiscard]] std::size_t padding()      const noexcept { return pad_; }

    [[nodiscard]] const Matrix& weights()      const noexcept { return weights_; }
    [[nodiscard]] const Matrix& bias()         const noexcept { return bias_; }
    [[nodiscard]] const Matrix& grad_weights() const noexcept { return grad_weights_; }
    [[nodiscard]] const Matrix& grad_bias()    const noexcept { return grad_bias_; }

    void set_weights(Matrix w);
    void set_bias(Matrix b);

private:
    std::size_t in_, out_, K_, H_, W_, H_out_, W_out_, pad_;

    Matrix weights_;       // (C_out, C_in * K * K)
    Matrix bias_;          // (C_out, 1)
    Matrix grad_weights_;  // (C_out, C_in * K * K)
    Matrix grad_bias_;     // (C_out, 1)
    Matrix col_cache_;     // (C_in*K*K, H_out*W_out*N) cached by forward_train
    std::size_t N_cache_{0};

    Matrix m_w_, v_w_, m_b_, v_b_;  // Adam moments (zero-sized until first step)
};

} // namespace neuronix
