#include "neuronix/losses/mse_loss.hpp"

#include <stdexcept>

namespace neuronix {

double MSELoss::forward(const Matrix& predicted, const Matrix& target) {
    if (predicted.rows() != target.rows() || predicted.cols() != target.cols()) {
        throw std::invalid_argument{"MSELoss: predicted and target shapes must match"};
    }
    Matrix diff = predicted - target;
    const double n = static_cast<double>(predicted.size());
    grad_ = (2.0 / n) * diff;
    return diff.hadamard(diff).sum() / n;
}

} // namespace neuronix
