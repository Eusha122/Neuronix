#include "neuronix/layers/conv2d.hpp"

#include <cmath>
#include <stdexcept>

namespace neuronix {

// ── im2col / col2im helpers ───────────────────────────────────────────────────

namespace {

// input:  (C * H * W, N)
// output: (C * K * K, H_out * W_out * N)   — one column per (sample, output position)
Matrix im2col(const Matrix& input,
              std::size_t N,
              std::size_t C, std::size_t H, std::size_t W,
              std::size_t K, std::size_t pad,
              std::size_t H_out, std::size_t W_out)
{
    Matrix col = Matrix::zeros(C * K * K, H_out * W_out * N);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t ho = 0; ho < H_out; ++ho) {
            for (std::size_t wo = 0; wo < W_out; ++wo) {
                const std::size_t col_c = n * H_out * W_out + ho * W_out + wo;
                for (std::size_t c = 0; c < C; ++c) {
                    for (std::size_t kh = 0; kh < K; ++kh) {
                        for (std::size_t kw = 0; kw < K; ++kw) {
                            const std::size_t col_r = (c * K + kh) * K + kw;
                            const long long hi = static_cast<long long>(ho + kh)
                                               - static_cast<long long>(pad);
                            const long long wi = static_cast<long long>(wo + kw)
                                               - static_cast<long long>(pad);
                            if (hi >= 0 && hi < static_cast<long long>(H) &&
                                wi >= 0 && wi < static_cast<long long>(W)) {
                                const std::size_t in_r =
                                    c * H * W +
                                    static_cast<std::size_t>(hi) * W +
                                    static_cast<std::size_t>(wi);
                                col(col_r, col_c) = input(in_r, n);
                            }
                        }
                    }
                }
            }
        }
    }
    return col;
}

// col2im: adjoint of im2col — accumulates into input gradient
// col:    (C * K * K, H_out * W_out * N)
// output: (C * H * W, N)
Matrix col2im(const Matrix& col,
              std::size_t N,
              std::size_t C, std::size_t H, std::size_t W,
              std::size_t K, std::size_t pad,
              std::size_t H_out, std::size_t W_out)
{
    Matrix grad = Matrix::zeros(C * H * W, N);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t ho = 0; ho < H_out; ++ho) {
            for (std::size_t wo = 0; wo < W_out; ++wo) {
                const std::size_t col_c = n * H_out * W_out + ho * W_out + wo;
                for (std::size_t c = 0; c < C; ++c) {
                    for (std::size_t kh = 0; kh < K; ++kh) {
                        for (std::size_t kw = 0; kw < K; ++kw) {
                            const std::size_t col_r = (c * K + kh) * K + kw;
                            const long long hi = static_cast<long long>(ho + kh)
                                               - static_cast<long long>(pad);
                            const long long wi = static_cast<long long>(wo + kw)
                                               - static_cast<long long>(pad);
                            if (hi >= 0 && hi < static_cast<long long>(H) &&
                                wi >= 0 && wi < static_cast<long long>(W)) {
                                const std::size_t in_r =
                                    c * H * W +
                                    static_cast<std::size_t>(hi) * W +
                                    static_cast<std::size_t>(wi);
                                grad(in_r, n) += col(col_r, col_c);
                            }
                        }
                    }
                }
            }
        }
    }
    return grad;
}

// Repack (C_out, H_out*W_out*N) → (C_out*H_out*W_out, N)
Matrix pack_output(const Matrix& m,
                   std::size_t C_out,
                   std::size_t H_out, std::size_t W_out,
                   std::size_t N)
{
    const std::size_t spatial = H_out * W_out;
    Matrix out = Matrix::zeros(C_out * spatial, N);
    for (std::size_t co = 0; co < C_out; ++co)
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t s = 0; s < spatial; ++s)
                out(co * spatial + s, n) = m(co, n * spatial + s);
    return out;
}

// Unpack (C_out*H_out*W_out, N) → (C_out, H_out*W_out*N)
Matrix unpack_grad(const Matrix& g,
                   std::size_t C_out,
                   std::size_t H_out, std::size_t W_out,
                   std::size_t N)
{
    const std::size_t spatial = H_out * W_out;
    Matrix out = Matrix::zeros(C_out, spatial * N);
    for (std::size_t r = 0; r < g.rows(); ++r) {
        const std::size_t co = r / spatial;
        const std::size_t s  = r % spatial;
        for (std::size_t n = 0; n < N; ++n)
            out(co, n * spatial + s) = g(r, n);
    }
    return out;
}

} // anonymous namespace

// ── Conv2D ────────────────────────────────────────────────────────────────────

Conv2D::Conv2D(std::size_t in_channels,
               std::size_t out_channels,
               std::size_t kernel_size,
               std::size_t in_height,
               std::size_t in_width,
               std::size_t padding)
    : in_{in_channels}, out_{out_channels}, K_{kernel_size},
      H_{in_height}, W_{in_width}, pad_{padding}
{
    if (in_ == 0 || out_ == 0 || K_ == 0)
        throw std::invalid_argument{"Conv2D: channels and kernel_size must be > 0"};
    if (H_ + 2 * pad_ < K_ || W_ + 2 * pad_ < K_)
        throw std::invalid_argument{"Conv2D: kernel larger than padded input"};

    H_out_ = H_ + 2 * pad_ - K_ + 1;
    W_out_ = W_ + 2 * pad_ - K_ + 1;

    weights_      = Matrix::he_normal(out_, in_ * K_ * K_);
    bias_         = Matrix::zeros(out_, 1);
    grad_weights_ = Matrix::zeros(out_, in_ * K_ * K_);
    grad_bias_    = Matrix::zeros(out_, 1);
}

Matrix Conv2D::forward(const Matrix& input) const {
    const auto N = input.cols();
    Matrix col     = im2col(input, N, in_, H_, W_, K_, pad_, H_out_, W_out_);
    Matrix col_out = weights_ * col;   // (C_out, H_out*W_out*N)
    for (std::size_t c = 0; c < col_out.cols(); ++c)
        for (std::size_t r = 0; r < out_; ++r)
            col_out(r, c) += bias_(r, 0);
    return pack_output(col_out, out_, H_out_, W_out_, N);
}

Matrix Conv2D::forward_train(const Matrix& input) {
    const auto N  = input.cols();
    N_cache_      = N;
    col_cache_    = im2col(input, N, in_, H_, W_, K_, pad_, H_out_, W_out_);
    Matrix col_out = weights_ * col_cache_;
    for (std::size_t c = 0; c < col_out.cols(); ++c)
        for (std::size_t r = 0; r < out_; ++r)
            col_out(r, c) += bias_(r, 0);
    return pack_output(col_out, out_, H_out_, W_out_, N);
}

Matrix Conv2D::backward(const Matrix& grad_output) {
    Matrix g = unpack_grad(grad_output, out_, H_out_, W_out_, N_cache_);

    // dL/dW += g * col_cache^T
    grad_weights_ += g * col_cache_.transpose();

    // dL/db += row sums of g
    for (std::size_t r = 0; r < out_; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < g.cols(); ++c) s += g(r, c);
        grad_bias_(r, 0) += s;
    }

    // dL/dX = col2im( W^T * g )
    return col2im(weights_.transpose() * g,
                  N_cache_, in_, H_, W_, K_, pad_, H_out_, W_out_);
}

void Conv2D::update(double lr) {
    weights_ -= lr * grad_weights_;
    bias_    -= lr * grad_bias_;
}

void Conv2D::adam_step(double lr, double beta1, double beta2,
                       double eps, std::size_t t) {
    if (m_w_.rows() == 0) {
        m_w_ = Matrix::zeros(out_, in_ * K_ * K_);
        v_w_ = Matrix::zeros(out_, in_ * K_ * K_);
        m_b_ = Matrix::zeros(out_, 1);
        v_b_ = Matrix::zeros(out_, 1);
    }
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
    for (std::size_t r = 0; r < out_; ++r) {
        for (std::size_t c = 0; c < in_ * K_ * K_; ++c) {
            const double gw = grad_weights_(r, c);
            m_w_(r, c) = beta1 * m_w_(r, c) + (1.0 - beta1) * gw;
            v_w_(r, c) = beta2 * v_w_(r, c) + (1.0 - beta2) * gw * gw;
            weights_(r, c) -= lr * (m_w_(r, c) / bc1) /
                              (std::sqrt(v_w_(r, c) / bc2) + eps);
        }
        const double gb = grad_bias_(r, 0);
        m_b_(r, 0) = beta1 * m_b_(r, 0) + (1.0 - beta1) * gb;
        v_b_(r, 0) = beta2 * v_b_(r, 0) + (1.0 - beta2) * gb * gb;
        bias_(r, 0) -= lr * (m_b_(r, 0) / bc1) /
                       (std::sqrt(v_b_(r, 0) / bc2) + eps);
    }
}

void Conv2D::zero_grad() {
    grad_weights_ = Matrix::zeros(out_, in_ * K_ * K_);
    grad_bias_    = Matrix::zeros(out_, 1);
}

std::size_t Conv2D::param_count() const noexcept {
    return out_ * in_ * K_ * K_ + out_;
}

void Conv2D::set_weights(Matrix w) {
    if (w.rows() != out_ || w.cols() != in_ * K_ * K_)
        throw std::invalid_argument{"Conv2D::set_weights: shape mismatch"};
    weights_ = std::move(w);
}

void Conv2D::set_bias(Matrix b) {
    if (b.rows() != out_ || b.cols() != 1)
        throw std::invalid_argument{"Conv2D::set_bias: shape mismatch"};
    bias_ = std::move(b);
}

} // namespace neuronix
