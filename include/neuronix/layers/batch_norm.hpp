#pragma once

#include <cstddef>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// Batch Normalization layer (Ioffe & Szegedy, 2015).
//
// forward()       — inference: uses running_mean / running_var
// forward_train() — training:  normalizes over the mini-batch,
//                              updates running statistics
//
// Learnable parameters: gamma (scale) and beta (shift), both shape (F, 1).
// Non-learnable:        running_mean, running_var, both shape (F, 1).
class BatchNorm : public Layer {
public:
    explicit BatchNorm(std::size_t num_features,
                       double eps      = 1e-5,
                       double momentum = 0.1);

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update(double lr) override;
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;
    void zero_grad() override;

    [[nodiscard]] std::string_view name()        const noexcept override { return "BatchNorm"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override { return 2 * F_; }

    [[nodiscard]] std::size_t num_features() const noexcept { return F_; }
    [[nodiscard]] double      eps()          const noexcept { return eps_; }
    [[nodiscard]] double      momentum()     const noexcept { return mom_; }

    [[nodiscard]] const Matrix& gamma()        const noexcept { return gamma_; }
    [[nodiscard]] const Matrix& beta()         const noexcept { return beta_; }
    [[nodiscard]] const Matrix& running_mean() const noexcept { return run_mean_; }
    [[nodiscard]] const Matrix& running_var()  const noexcept { return run_var_; }
    [[nodiscard]] const Matrix& grad_gamma()   const noexcept { return grad_gamma_; }
    [[nodiscard]] const Matrix& grad_beta()    const noexcept { return grad_beta_; }

    void set_gamma(Matrix g);
    void set_beta(Matrix b);
    void set_running_mean(Matrix m);
    void set_running_var(Matrix v);

private:
    std::size_t F_;     // num_features
    double      eps_;   // numerical stability
    double      mom_;   // momentum for running stats

    Matrix gamma_;      // (F, 1)  scale,  init 1
    Matrix beta_;       // (F, 1)  shift,  init 0
    Matrix run_mean_;   // (F, 1)  running mean, init 0
    Matrix run_var_;    // (F, 1)  running var,  init 1

    Matrix grad_gamma_; // (F, 1)
    Matrix grad_beta_;  // (F, 1)

    // Cached by forward_train for use in backward
    Matrix   x_hat_;     // (F, N) normalised input
    Matrix   std_inv_;   // (F, 1) 1/sqrt(var + eps)
    std::size_t N_{0};

    // Adam moments (lazy init)
    Matrix m_g_, v_g_, m_b_, v_b_;
};

} // namespace neuronix
