#include "neuronix/layers/batch_norm.hpp"

#include <cmath>
#include <stdexcept>

namespace neuronix {

BatchNorm::BatchNorm(std::size_t num_features, double eps, double momentum)
    : F_{num_features}, eps_{eps}, mom_{momentum},
      gamma_{Matrix::ones(F_, 1)},
      beta_{Matrix::zeros(F_, 1)},
      run_mean_{Matrix::zeros(F_, 1)},
      run_var_{Matrix::ones(F_, 1)},
      grad_gamma_{Matrix::zeros(F_, 1)},
      grad_beta_{Matrix::zeros(F_, 1)}
{
    if (F_ == 0)
        throw std::invalid_argument{"BatchNorm: num_features must be > 0"};
    if (eps_ <= 0.0)
        throw std::invalid_argument{"BatchNorm: eps must be > 0"};
    if (mom_ <= 0.0 || mom_ > 1.0)
        throw std::invalid_argument{"BatchNorm: momentum must be in (0, 1]"};
}

// ── Inference ─────────────────────────────────────────────────────────────────

Matrix BatchNorm::forward(const Matrix& input) const {
    Matrix out = Matrix::zeros(F_, input.cols());
    for (std::size_t r = 0; r < F_; ++r) {
        const double si = 1.0 / std::sqrt(run_var_(r, 0) + eps_);
        for (std::size_t c = 0; c < input.cols(); ++c)
            out(r, c) = gamma_(r, 0) * (input(r, c) - run_mean_(r, 0)) * si + beta_(r, 0);
    }
    return out;
}

// ── Training ──────────────────────────────────────────────────────────────────

Matrix BatchNorm::forward_train(const Matrix& input) {
    const std::size_t N = input.cols();
    N_ = N;
    const double Nd = static_cast<double>(N);

    // Per-feature batch mean
    Matrix batch_mean = Matrix::zeros(F_, 1);
    for (std::size_t r = 0; r < F_; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < N; ++c) s += input(r, c);
        batch_mean(r, 0) = s / Nd;
    }

    // Per-feature batch variance
    Matrix batch_var = Matrix::zeros(F_, 1);
    for (std::size_t r = 0; r < F_; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < N; ++c) {
            double d = input(r, c) - batch_mean(r, 0);
            s += d * d;
        }
        batch_var(r, 0) = s / Nd;
    }

    // Normalize, scale, shift — cache x_hat and std_inv for backward
    std_inv_ = Matrix::zeros(F_, 1);
    x_hat_   = Matrix::zeros(F_, N);
    Matrix out = Matrix::zeros(F_, N);
    for (std::size_t r = 0; r < F_; ++r) {
        std_inv_(r, 0) = 1.0 / std::sqrt(batch_var(r, 0) + eps_);
        for (std::size_t c = 0; c < N; ++c) {
            x_hat_(r, c) = (input(r, c) - batch_mean(r, 0)) * std_inv_(r, 0);
            out(r, c)    = gamma_(r, 0) * x_hat_(r, c) + beta_(r, 0);
        }
    }

    // Update running statistics with exponential moving average
    for (std::size_t r = 0; r < F_; ++r) {
        run_mean_(r, 0) = (1.0 - mom_) * run_mean_(r, 0) + mom_ * batch_mean(r, 0);
        run_var_(r, 0)  = (1.0 - mom_) * run_var_(r, 0)  + mom_ * batch_var(r, 0);
    }

    return out;
}

// ── Backward ──────────────────────────────────────────────────────────────────
//
// Given dL/dy, the full BatchNorm backward through the batch statistics is:
//
//   dL/d_gamma_r = sum_n( dL/dy_rn * x_hat_rn )
//   dL/d_beta_r  = sum_n( dL/dy_rn )
//
//   dL/dx_hat_rn = dL/dy_rn * gamma_r
//
//   dL/dx_rn = std_inv_r / N *
//              ( N * dL/dx_hat_rn
//                - sum_n(dL/dx_hat_rn)
//                - x_hat_rn * sum_n(dL/dx_hat_rn * x_hat_rn) )

Matrix BatchNorm::backward(const Matrix& grad_output) {
    const double Nd = static_cast<double>(N_);

    // Accumulate param gradients
    for (std::size_t r = 0; r < F_; ++r) {
        double dg = 0.0, db = 0.0;
        for (std::size_t c = 0; c < N_; ++c) {
            dg += grad_output(r, c) * x_hat_(r, c);
            db += grad_output(r, c);
        }
        grad_gamma_(r, 0) += dg;
        grad_beta_(r, 0)  += db;
    }

    // Input gradient
    Matrix grad_in = Matrix::zeros(F_, N_);
    for (std::size_t r = 0; r < F_; ++r) {
        double sum_dxh = 0.0, sum_dxh_xh = 0.0;
        for (std::size_t c = 0; c < N_; ++c) {
            const double dxh = grad_output(r, c) * gamma_(r, 0);
            sum_dxh    += dxh;
            sum_dxh_xh += dxh * x_hat_(r, c);
        }
        const double si = std_inv_(r, 0);
        for (std::size_t c = 0; c < N_; ++c) {
            const double dxh = grad_output(r, c) * gamma_(r, 0);
            grad_in(r, c) = (si / Nd) *
                (Nd * dxh - sum_dxh - x_hat_(r, c) * sum_dxh_xh);
        }
    }
    return grad_in;
}

// ── Parameter update ──────────────────────────────────────────────────────────

void BatchNorm::update(double lr) {
    gamma_ -= lr * grad_gamma_;
    beta_  -= lr * grad_beta_;
}

void BatchNorm::adam_step(double lr, double beta1, double beta2,
                           double eps_adam, std::size_t t) {
    if (m_g_.rows() == 0) {
        m_g_ = Matrix::zeros(F_, 1);
        v_g_ = Matrix::zeros(F_, 1);
        m_b_ = Matrix::zeros(F_, 1);
        v_b_ = Matrix::zeros(F_, 1);
    }
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
    for (std::size_t r = 0; r < F_; ++r) {
        const double gg = grad_gamma_(r, 0);
        m_g_(r, 0) = beta1 * m_g_(r, 0) + (1.0 - beta1) * gg;
        v_g_(r, 0) = beta2 * v_g_(r, 0) + (1.0 - beta2) * gg * gg;
        gamma_(r, 0) -= lr * (m_g_(r, 0) / bc1) /
                        (std::sqrt(v_g_(r, 0) / bc2) + eps_adam);

        const double gb = grad_beta_(r, 0);
        m_b_(r, 0) = beta1 * m_b_(r, 0) + (1.0 - beta1) * gb;
        v_b_(r, 0) = beta2 * v_b_(r, 0) + (1.0 - beta2) * gb * gb;
        beta_(r, 0) -= lr * (m_b_(r, 0) / bc1) /
                       (std::sqrt(v_b_(r, 0) / bc2) + eps_adam);
    }
}

void BatchNorm::zero_grad() {
    grad_gamma_ = Matrix::zeros(F_, 1);
    grad_beta_  = Matrix::zeros(F_, 1);
}

// ── Setters (for deserialization) ─────────────────────────────────────────────

void BatchNorm::set_gamma(Matrix g) {
    if (g.rows() != F_ || g.cols() != 1)
        throw std::invalid_argument{"BatchNorm::set_gamma: shape mismatch"};
    gamma_ = std::move(g);
}
void BatchNorm::set_beta(Matrix b) {
    if (b.rows() != F_ || b.cols() != 1)
        throw std::invalid_argument{"BatchNorm::set_beta: shape mismatch"};
    beta_ = std::move(b);
}
void BatchNorm::set_running_mean(Matrix m) {
    if (m.rows() != F_ || m.cols() != 1)
        throw std::invalid_argument{"BatchNorm::set_running_mean: shape mismatch"};
    run_mean_ = std::move(m);
}
void BatchNorm::set_running_var(Matrix v) {
    if (v.rows() != F_ || v.cols() != 1)
        throw std::invalid_argument{"BatchNorm::set_running_var: shape mismatch"};
    run_var_ = std::move(v);
}

} // namespace neuronix
