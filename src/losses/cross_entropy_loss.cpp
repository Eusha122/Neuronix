#include "neuronix/losses/cross_entropy_loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuronix {

double CrossEntropyLoss::forward(const Matrix& logits, const Matrix& target) {
    if (logits.rows() != target.rows() || logits.cols() != target.cols()) {
        throw std::invalid_argument{
            "CrossEntropyLoss: logits and target shapes must match"};
    }

    const std::size_t out   = logits.rows();
    const std::size_t batch = logits.cols();

    // Numerically-stable softmax per column
    Matrix probs = logits;
    for (std::size_t c = 0; c < batch; ++c) {
        double max_val = probs(0, c);
        for (std::size_t r = 1; r < out; ++r)
            max_val = std::max(max_val, probs(r, c));

        double sum = 0.0;
        for (std::size_t r = 0; r < out; ++r) {
            probs(r, c) = std::exp(probs(r, c) - max_val);
            sum += probs(r, c);
        }
        for (std::size_t r = 0; r < out; ++r)
            probs(r, c) /= sum;
    }

    // Cross-entropy: -mean(sum_r(target * log(probs)))
    double loss = 0.0;
    for (std::size_t c = 0; c < batch; ++c)
        for (std::size_t r = 0; r < out; ++r)
            loss -= target(r, c) * std::log(probs(r, c) + 1e-15);
    loss /= static_cast<double>(batch);

    // Combined softmax+CE gradient: (probs - target) / batch
    grad_ = (1.0 / static_cast<double>(batch)) * (probs - target);
    return loss;
}

} // namespace neuronix
