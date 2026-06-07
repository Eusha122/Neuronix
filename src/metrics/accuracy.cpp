#include "neuronix/metrics/accuracy.hpp"

#include <stdexcept>

namespace neuronix::metrics {

static std::size_t argmax_col(const Matrix& m, std::size_t col) {
    std::size_t best = 0;
    for (std::size_t r = 1; r < m.rows(); ++r)
        if (m(r, col) > m(best, col)) best = r;
    return best;
}

double accuracy(const Matrix& predicted, const std::vector<int>& true_labels) {
    if (predicted.cols() != true_labels.size()) {
        throw std::invalid_argument{
            "accuracy: predicted columns must equal true_labels size"};
    }
    std::size_t correct = 0;
    for (std::size_t c = 0; c < predicted.cols(); ++c)
        if (static_cast<int>(argmax_col(predicted, c)) == true_labels[c])
            ++correct;
    return static_cast<double>(correct) / static_cast<double>(predicted.cols());
}

double accuracy(const Matrix& predicted, const Matrix& one_hot) {
    if (predicted.rows() != one_hot.rows() || predicted.cols() != one_hot.cols()) {
        throw std::invalid_argument{"accuracy: predicted and one_hot shapes must match"};
    }
    std::size_t correct = 0;
    for (std::size_t c = 0; c < predicted.cols(); ++c)
        if (argmax_col(predicted, c) == argmax_col(one_hot, c))
            ++correct;
    return static_cast<double>(correct) / static_cast<double>(predicted.cols());
}

} // namespace neuronix::metrics
