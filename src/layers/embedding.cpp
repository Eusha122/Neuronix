#include "neuronix/layers/embedding.hpp"

#include <cmath>
#include <stdexcept>

namespace neuronix {

Embedding::Embedding(std::size_t vocab_size,
                       std::size_t embed_dim,
                       std::size_t seq_len)
    : V_{vocab_size}, E_{embed_dim}, T_{seq_len}
    , table_{Matrix::random_normal(V_, E_, 0.0, 0.1)}
    , grad_table_{Matrix::zeros(V_, E_)}
{}

// Lookup embeddings for all (t, n) positions
static Matrix lookup(const Matrix& table,
                     const Matrix& indices,
                     std::size_t E)
{
    const std::size_t T = indices.rows();
    const std::size_t N = indices.cols();
    Matrix out = Matrix::zeros(T * E, N);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) {
            const auto idx = static_cast<std::size_t>(indices(t, n));
            for (std::size_t e = 0; e < E; ++e)
                out(t * E + e, n) = table(idx, e);
        }
    return out;
}

Matrix Embedding::forward(const Matrix& input) const {
    return lookup(table_, input, E_);
}

Matrix Embedding::forward_train(const Matrix& input) {
    idx_cache_ = input;
    return lookup(table_, input, E_);
}

Matrix Embedding::backward(const Matrix& grad_output) {
    const std::size_t N = grad_output.cols();
    // Scatter-add gradient to embedding table rows
    for (std::size_t t = 0; t < T_; ++t)
        for (std::size_t n = 0; n < N; ++n) {
            const auto idx = static_cast<std::size_t>(idx_cache_(t, n));
            for (std::size_t e = 0; e < E_; ++e)
                grad_table_(idx, e) += grad_output(t * E_ + e, n);
        }
    // Embedding layers don't propagate gradients to their integer inputs
    return Matrix::zeros(T_, N);
}

void Embedding::zero_grad() {
    grad_table_ = Matrix::zeros(V_, E_);
}

void Embedding::update(double lr) {
    table_ -= grad_table_ * lr;
}

void Embedding::adam_step(double lr, double beta1, double beta2,
                            double eps, std::size_t t) {
    if (m_.empty()) {
        m_ = Matrix::zeros(V_, E_);
        v_ = Matrix::zeros(V_, E_);
    }
    const double bc1  = 1.0 - std::pow(beta1, static_cast<double>(t));
    const double bc2  = 1.0 - std::pow(beta2, static_cast<double>(t));
    const double lr_t = lr * std::sqrt(bc2) / bc1;

    m_ = m_ * beta1 + grad_table_ * (1.0 - beta1);
    v_ = v_ * beta2 + grad_table_.hadamard(grad_table_) * (1.0 - beta2);
    table_ -= (m_ * lr_t).hadamard(
        v_.apply([eps](double vi){ return 1.0 / (std::sqrt(vi) + eps); }));
}

void Embedding::adamw_step(double lr, double beta1, double beta2,
                             double eps, double wd, std::size_t t) {
    table_ *= (1.0 - lr * wd);
    adam_step(lr, beta1, beta2, eps, t);
}

void Embedding::set_weights(Matrix w) { table_ = std::move(w); }

} // namespace neuronix
