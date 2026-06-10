#include "neuronix/layers/lstm.hpp"

#include <cmath>
#include <stdexcept>

namespace neuronix {

namespace {
    inline double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }
    inline double sigmoid_d(double s) { return s * (1.0 - s); }   // s = sigmoid(x)
    inline double tanh_d(double t) { return 1.0 - t * t; }         // t = tanh(x)
}

// ── Construction ─────────────────────────────────────────────────────────────

LSTM::LSTM(std::size_t input_size, std::size_t hidden_size, std::size_t seq_len)
    : I_{input_size}, H_{hidden_size}, T_{seq_len}
{
    const double std_x = 1.0 / std::sqrt(static_cast<double>(I_));
    const double std_h = 1.0 / std::sqrt(static_cast<double>(H_));

    W_x_ = Matrix::random_normal(4 * H_, I_, 0.0, std_x);
    W_h_ = Matrix::random_normal(4 * H_, H_, 0.0, std_h);
    b_   = Matrix::zeros(4 * H_, 1);

    // Initialise forget-gate bias to 1 — helps remember early in training
    for (std::size_t h = 0; h < H_; ++h)
        b_(H_ + h, 0) = 1.0;

    grad_W_x_ = Matrix::zeros(4 * H_, I_);
    grad_W_h_ = Matrix::zeros(4 * H_, H_);
    grad_b_   = Matrix::zeros(4 * H_, 1);
}

// ── One step forward ─────────────────────────────────────────────────────────

void LSTM::step_forward(const Matrix& x_t,
                         const Matrix& h_prev,
                         const Matrix& c_prev,
                         Matrix& h_new, Matrix& c_new,
                         Matrix& i_t, Matrix& f_t,
                         Matrix& g_t, Matrix& o_t,
                         Matrix& tanh_c) const
{
    const std::size_t N = x_t.cols();

    // gates = W_x*x + W_h*h + b  (broadcast bias across batch)
    Matrix gates = W_x_ * x_t + W_h_ * h_prev
                 + b_ * Matrix::ones(1, N);  // (4H, N)

    i_t = Matrix::zeros(H_, N);
    f_t = Matrix::zeros(H_, N);
    g_t = Matrix::zeros(H_, N);
    o_t = Matrix::zeros(H_, N);

    for (std::size_t h = 0; h < H_; ++h)
        for (std::size_t n = 0; n < N; ++n) {
            i_t(h, n) = sigmoid(gates(h,           n));
            f_t(h, n) = sigmoid(gates(H_      + h, n));
            g_t(h, n) = std::tanh(gates(2 * H_ + h, n));
            o_t(h, n) = sigmoid(gates(3 * H_ + h, n));
        }

    c_new  = f_t.hadamard(c_prev) + i_t.hadamard(g_t);
    tanh_c = c_new.apply([](double v){ return std::tanh(v); });
    h_new  = o_t.hadamard(tanh_c);
}

// ── Inference forward ────────────────────────────────────────────────────────

Matrix LSTM::forward(const Matrix& input) const {
    const std::size_t N = input.cols();
    Matrix h = Matrix::zeros(H_, N);
    Matrix c = Matrix::zeros(H_, N);

    for (std::size_t t = 0; t < T_; ++t) {
        // Extract x_t: rows [t*I_, (t+1)*I_)
        Matrix x_t = Matrix::zeros(I_, N);
        for (std::size_t i = 0; i < I_; ++i)
            for (std::size_t n = 0; n < N; ++n)
                x_t(i, n) = input(t * I_ + i, n);

        Matrix h_new, c_new, i_t, f_t, g_t, o_t, tanh_c;
        step_forward(x_t, h, c, h_new, c_new, i_t, f_t, g_t, o_t, tanh_c);
        h = std::move(h_new);
        c = std::move(c_new);
    }
    return h;
}

// ── Training forward ─────────────────────────────────────────────────────────

Matrix LSTM::forward_train(const Matrix& input) {
    const std::size_t N = input.cols();

    h_cache_.resize(T_ + 1);
    c_cache_.resize(T_ + 1);
    x_cache_.resize(T_);
    i_cache_.resize(T_);
    f_cache_.resize(T_);
    g_cache_.resize(T_);
    o_cache_.resize(T_);
    tanh_c_cache_.resize(T_);

    h_cache_[0] = Matrix::zeros(H_, N);
    c_cache_[0] = Matrix::zeros(H_, N);

    for (std::size_t t = 0; t < T_; ++t) {
        // Extract x_t
        Matrix x_t = Matrix::zeros(I_, N);
        for (std::size_t i = 0; i < I_; ++i)
            for (std::size_t n = 0; n < N; ++n)
                x_t(i, n) = input(t * I_ + i, n);
        x_cache_[t] = x_t;

        step_forward(x_t, h_cache_[t], c_cache_[t],
                     h_cache_[t + 1], c_cache_[t + 1],
                     i_cache_[t], f_cache_[t],
                     g_cache_[t], o_cache_[t], tanh_c_cache_[t]);
    }

    return h_cache_[T_];
}

// ── Backward (BPTT) ──────────────────────────────────────────────────────────

Matrix LSTM::backward(const Matrix& grad_h_last) {
    const std::size_t N = grad_h_last.cols();

    Matrix d_input = Matrix::zeros(T_ * I_, N);
    Matrix dh = grad_h_last;             // gradient of loss w.r.t. h_t
    Matrix dc = Matrix::zeros(H_, N);    // gradient w.r.t. cell state

    for (std::ptrdiff_t t = static_cast<std::ptrdiff_t>(T_) - 1; t >= 0; --t) {
        const std::size_t ut = static_cast<std::size_t>(t);

        const Matrix& i_t    = i_cache_[ut];
        const Matrix& f_t    = f_cache_[ut];
        const Matrix& g_t    = g_cache_[ut];
        const Matrix& o_t    = o_cache_[ut];
        const Matrix& tc     = tanh_c_cache_[ut];  // tanh(c_t)
        const Matrix& c_prev = c_cache_[ut];
        const Matrix& h_prev = h_cache_[ut];
        const Matrix& x_t    = x_cache_[ut];

        // Backward through h_t = o_t ⊙ tanh(c_t)
        Matrix d_o    = dh.hadamard(tc);
        Matrix d_tanh = dh.hadamard(o_t);

        // Backward through tanh(c_t)
        dc += d_tanh.hadamard(tc.apply(tanh_d));

        // Backward through c_t = f ⊙ c_prev + i ⊙ g
        Matrix d_f     = dc.hadamard(c_prev);
        Matrix d_i     = dc.hadamard(g_t);
        Matrix d_g     = dc.hadamard(i_t);
        Matrix dc_prev = dc.hadamard(f_t);

        // Backward through gate activations
        Matrix d_gates = Matrix::zeros(4 * H_, N);
        for (std::size_t h = 0; h < H_; ++h)
            for (std::size_t n = 0; n < N; ++n) {
                d_gates(h,           n) = d_i(h, n) * sigmoid_d(i_t(h, n));
                d_gates(H_      + h, n) = d_f(h, n) * sigmoid_d(f_t(h, n));
                d_gates(2 * H_ + h, n) = d_g(h, n) * tanh_d(g_t(h, n));
                d_gates(3 * H_ + h, n) = d_o(h, n) * sigmoid_d(o_t(h, n));
            }

        // Clip gate gradients to prevent explosion
        d_gates = d_gates.apply([](double v){
            return v > 5.0 ? 5.0 : (v < -5.0 ? -5.0 : v);
        });

        // Accumulate weight gradients
        grad_W_x_ += d_gates * x_t.transpose();
        grad_W_h_ += d_gates * h_prev.transpose();
        grad_b_   += d_gates * Matrix::ones(N, 1);

        // Gradient to input x_t
        Matrix dx_t = W_x_.transpose() * d_gates;
        for (std::size_t i = 0; i < I_; ++i)
            for (std::size_t n = 0; n < N; ++n)
                d_input(ut * I_ + i, n) = dx_t(i, n);

        // Gradient to previous hidden state
        dh = W_h_.transpose() * d_gates;
        dc = dc_prev;
    }

    return d_input;
}

// ── Parameter updates ─────────────────────────────────────────────────────────

void LSTM::zero_grad() {
    grad_W_x_ = Matrix::zeros(4 * H_, I_);
    grad_W_h_ = Matrix::zeros(4 * H_, H_);
    grad_b_   = Matrix::zeros(4 * H_, 1);
}

void LSTM::update(double lr) {
    W_x_ -= grad_W_x_ * lr;
    W_h_ -= grad_W_h_ * lr;
    b_   -= grad_b_   * lr;
}

void LSTM::adam_step(double lr, double beta1, double beta2,
                      double eps, std::size_t t) {
    if (m_Wx_.empty()) {
        m_Wx_ = Matrix::zeros(4 * H_, I_);
        v_Wx_ = Matrix::zeros(4 * H_, I_);
        m_Wh_ = Matrix::zeros(4 * H_, H_);
        v_Wh_ = Matrix::zeros(4 * H_, H_);
        m_b_  = Matrix::zeros(4 * H_, 1);
        v_b_  = Matrix::zeros(4 * H_, 1);
    }
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
    const double lr_t = lr * std::sqrt(bc2) / bc1;

    auto adam_update = [&](Matrix& param, Matrix& grad,
                           Matrix& m, Matrix& v) {
        m = m * beta1 + grad * (1.0 - beta1);
        v = v * beta2 + grad.hadamard(grad) * (1.0 - beta2);
        param -= (m * lr_t).hadamard(
            v.apply([eps](double vi){ return 1.0 / (std::sqrt(vi) + eps); }));
    };

    adam_update(W_x_, grad_W_x_, m_Wx_, v_Wx_);
    adam_update(W_h_, grad_W_h_, m_Wh_, v_Wh_);
    adam_update(b_,   grad_b_,   m_b_,  v_b_);
}

void LSTM::adamw_step(double lr, double beta1, double beta2,
                       double eps, double wd, std::size_t t) {
    W_x_ *= (1.0 - lr * wd);
    W_h_ *= (1.0 - lr * wd);
    adam_step(lr, beta1, beta2, eps, t);
}

std::size_t LSTM::param_count() const noexcept {
    return 4 * H_ * I_ + 4 * H_ * H_ + 4 * H_;
}

void LSTM::set_W_x(Matrix m) { W_x_ = std::move(m); }
void LSTM::set_W_h(Matrix m) { W_h_ = std::move(m); }
void LSTM::set_b  (Matrix m) { b_   = std::move(m); }

} // namespace neuronix
