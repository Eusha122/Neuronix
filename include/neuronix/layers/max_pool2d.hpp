#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// 2-D max-pooling layer (non-overlapping windows, stride = pool_size).
//
// Input layout : (C * H     * W,     N)
// Output layout: (C * H_out * W_out, N)
//
// H_out = H / pool_size  (H must be divisible by pool_size)
// W_out = W / pool_size  (W must be divisible by pool_size)
class MaxPool2D : public Layer {
public:
    MaxPool2D(std::size_t pool_size,
              std::size_t channels,
              std::size_t in_height,
              std::size_t in_width)
        : P_{pool_size}, C_{channels}, H_{in_height}, W_{in_width},
          H_out_{H_ / P_}, W_out_{W_ / P_}
    {
        if (P_ == 0)
            throw std::invalid_argument{"MaxPool2D: pool_size must be > 0"};
        if (H_ % P_ != 0 || W_ % P_ != 0)
            throw std::invalid_argument{
                "MaxPool2D: input H and W must be divisible by pool_size"};
    }

    // Inference: pure forward, no mask
    [[nodiscard]] Matrix forward(const Matrix& input) const override {
        const std::size_t N = input.cols();
        Matrix out = Matrix::zeros(C_ * H_out_ * W_out_, N);
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t c = 0; c < C_; ++c)
                for (std::size_t ho = 0; ho < H_out_; ++ho)
                    for (std::size_t wo = 0; wo < W_out_; ++wo) {
                        double mx = -std::numeric_limits<double>::infinity();
                        for (std::size_t ph = 0; ph < P_; ++ph)
                            for (std::size_t pw = 0; pw < P_; ++pw)
                                mx = std::max(mx,
                                    input(c * H_ * W_ + (ho*P_+ph)*W_ + (wo*P_+pw), n));
                        out(c * H_out_ * W_out_ + ho * W_out_ + wo, n) = mx;
                    }
        return out;
    }

    // Training: forward + build argmax mask for backward
    [[nodiscard]] Matrix forward_train(const Matrix& input) override {
        const std::size_t N = input.cols();
        mask_ = Matrix::zeros(C_ * H_ * W_, N);
        Matrix out = Matrix::zeros(C_ * H_out_ * W_out_, N);
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t c = 0; c < C_; ++c)
                for (std::size_t ho = 0; ho < H_out_; ++ho)
                    for (std::size_t wo = 0; wo < W_out_; ++wo) {
                        double mx = -std::numeric_limits<double>::infinity();
                        std::size_t mx_r = c * H_ * W_ + ho * P_ * W_ + wo * P_;
                        for (std::size_t ph = 0; ph < P_; ++ph)
                            for (std::size_t pw = 0; pw < P_; ++pw) {
                                std::size_t in_r = c*H_*W_ + (ho*P_+ph)*W_ + (wo*P_+pw);
                                if (input(in_r, n) > mx) { mx = input(in_r, n); mx_r = in_r; }
                            }
                        out(c * H_out_ * W_out_ + ho * W_out_ + wo, n) = mx;
                        mask_(mx_r, n) = 1.0;
                    }
        return out;
    }

    // Backward: route gradient only through max positions
    Matrix backward(const Matrix& grad_output) override {
        const std::size_t N = grad_output.cols();
        Matrix grad_in = Matrix::zeros(C_ * H_ * W_, N);
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t c = 0; c < C_; ++c)
                for (std::size_t ho = 0; ho < H_out_; ++ho)
                    for (std::size_t wo = 0; wo < W_out_; ++wo) {
                        const double g = grad_output(
                            c * H_out_ * W_out_ + ho * W_out_ + wo, n);
                        for (std::size_t ph = 0; ph < P_; ++ph)
                            for (std::size_t pw = 0; pw < P_; ++pw) {
                                std::size_t in_r = c*H_*W_ + (ho*P_+ph)*W_ + (wo*P_+pw);
                                grad_in(in_r, n) += mask_(in_r, n) * g;
                            }
                    }
        return grad_in;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "MaxPool2D"; }

    [[nodiscard]] std::size_t pool_size()  const noexcept { return P_; }
    [[nodiscard]] std::size_t channels()   const noexcept { return C_; }
    [[nodiscard]] std::size_t in_height()  const noexcept { return H_; }
    [[nodiscard]] std::size_t in_width()   const noexcept { return W_; }
    [[nodiscard]] std::size_t out_height() const noexcept { return H_out_; }
    [[nodiscard]] std::size_t out_width()  const noexcept { return W_out_; }

private:
    std::size_t P_, C_, H_, W_, H_out_, W_out_;
    Matrix mask_;
};

} // namespace neuronix
