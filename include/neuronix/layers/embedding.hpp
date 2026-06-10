#pragma once

#include <cstddef>
#include <string_view>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// Embedding lookup layer — maps integer token indices to dense vectors.
//
// Input layout : (seq_len, N)               — each entry is a token index [0, vocab)
// Output layout: (seq_len * embed_dim, N)   — concatenated embedding vectors
//
// The embedding table has shape (vocab_size, embed_dim) stored row-major.
// Gradient is accumulated densely (same index updated multiple times sums correctly).
class Embedding : public Layer {
public:
    Embedding(std::size_t vocab_size,
              std::size_t embed_dim,
              std::size_t seq_len);

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;

    void update(double lr) override;
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;
    void adamw_step(double lr, double beta1, double beta2,
                    double eps, double wd, std::size_t t) override;
    void zero_grad() override;

    [[nodiscard]] std::string_view name()        const noexcept override { return "Embedding"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override { return V_ * E_; }

    [[nodiscard]] std::size_t vocab_size() const noexcept { return V_; }
    [[nodiscard]] std::size_t embed_dim()  const noexcept { return E_; }
    [[nodiscard]] std::size_t seq_len()    const noexcept { return T_; }
    [[nodiscard]] const Matrix& weights()  const noexcept { return table_; }
    void set_weights(Matrix w);

private:
    std::size_t V_, E_, T_;

    Matrix table_;       // (V, E) — embedding table
    Matrix grad_table_;  // (V, E) — accumulated gradients
    Matrix m_, v_;       // Adam moments

    Matrix idx_cache_;   // (T, N) — token indices saved for backward
};

} // namespace neuronix
