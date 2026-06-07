#include "neuronix/layers/dense.hpp"

#include <stdexcept>

namespace neuronix {

Dense::Dense(std::size_t in_features, std::size_t out_features)
    : in_{in_features},
      out_{out_features},
      weights_{Matrix::he_normal(out_features, in_features)},
      bias_{Matrix::zeros(out_features, 1)} {}

Matrix Dense::forward(const Matrix& input) const {
    if (input.rows() != in_) {
        throw std::invalid_argument{
            "Dense::forward: input row count does not match in_features"};
    }
    Matrix result = weights_ * input;  // (out, batch_size)
    for (std::size_t c = 0; c < result.cols(); ++c)
        for (std::size_t r = 0; r < result.rows(); ++r)
            result(r, c) += bias_(r, 0);
    return result;
}

std::size_t Dense::param_count() const noexcept {
    return weights_.size() + bias_.size();
}

void Dense::set_weights(Matrix w) {
    if (w.rows() != out_ || w.cols() != in_) {
        throw std::invalid_argument{"Dense::set_weights: shape mismatch"};
    }
    weights_ = std::move(w);
}

void Dense::set_bias(Matrix b) {
    if (b.rows() != out_ || b.cols() != 1) {
        throw std::invalid_argument{"Dense::set_bias: shape mismatch"};
    }
    bias_ = std::move(b);
}

} // namespace neuronix
