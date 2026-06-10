#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

// Single-layer LSTM — processes a fixed-length sequence and returns the
// last hidden state.
//
// Input layout : (seq_len * input_size, N)  — time steps stacked row-wise
// Output layout: (hidden_size, N)           — h at the last time step
//
// Gates (combined weight matrix for efficiency):
//   [i, f, g, o] = W_x * x_t + W_h * h_{t-1} + b
//   i = sigmoid(...)    input gate
//   f = sigmoid(...)    forget gate
//   g = tanh(...)       cell gate
//   o = sigmoid(...)    output gate
//   c_t = f ⊙ c_{t-1} + i ⊙ g
//   h_t = o ⊙ tanh(c_t)
class LSTM : public Layer {
public:
    LSTM(std::size_t input_size,
         std::size_t hidden_size,
         std::size_t seq_len);

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] Matrix forward_train(const Matrix& input) override;
    Matrix backward(const Matrix& grad_h_last) override;

    void update(double lr) override;
    void adam_step(double lr, double beta1, double beta2,
                   double eps, std::size_t t) override;
    void adamw_step(double lr, double beta1, double beta2,
                    double eps, double wd, std::size_t t) override;
    void zero_grad() override;

    [[nodiscard]] std::string_view name()        const noexcept override { return "LSTM"; }
    [[nodiscard]] std::size_t      param_count() const noexcept override;

    // Accessors for serialization
    [[nodiscard]] std::size_t input_size()  const noexcept { return I_; }
    [[nodiscard]] std::size_t hidden_size() const noexcept { return H_; }
    [[nodiscard]] std::size_t seq_len()     const noexcept { return T_; }
    [[nodiscard]] const Matrix& W_x() const noexcept { return W_x_; }
    [[nodiscard]] const Matrix& W_h() const noexcept { return W_h_; }
    [[nodiscard]] const Matrix& b()   const noexcept { return b_;   }
    void set_W_x(Matrix m);
    void set_W_h(Matrix m);
    void set_b  (Matrix m);

private:
    std::size_t I_, H_, T_;   // input_size, hidden_size, seq_len

    Matrix W_x_;   // (4H, I)  — input weights  [i|f|g|o]
    Matrix W_h_;   // (4H, H)  — recurrent weights
    Matrix b_;     // (4H, 1)  — bias

    Matrix grad_W_x_, grad_W_h_, grad_b_;

    // Adam moments
    Matrix m_Wx_, v_Wx_, m_Wh_, v_Wh_, m_b_, v_b_;

    // Training cache (T+1 entries for h and c; T entries for gates)
    std::vector<Matrix> h_cache_;     // h_cache_[0]=zeros, h_cache_[t+1]=h_t
    std::vector<Matrix> c_cache_;     // c_cache_[0]=zeros, c_cache_[t+1]=c_t
    std::vector<Matrix> x_cache_;     // x_cache_[t] = x_t   (I, N)
    std::vector<Matrix> i_cache_;     // input gate  (H, N)
    std::vector<Matrix> f_cache_;     // forget gate (H, N)
    std::vector<Matrix> g_cache_;     // cell gate   (H, N)
    std::vector<Matrix> o_cache_;     // output gate (H, N)
    std::vector<Matrix> tanh_c_cache_; // tanh(c_t)  (H, N)

    // Run one forward step, storing gate activations
    void step_forward(const Matrix& x_t,
                      const Matrix& h_prev,
                      const Matrix& c_prev,
                      Matrix& h_new, Matrix& c_new,
                      Matrix& i_t, Matrix& f_t,
                      Matrix& g_t, Matrix& o_t,
                      Matrix& tanh_c) const;
};

} // namespace neuronix
