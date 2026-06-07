#include "neuronix/activations/softmax.hpp"

#include <algorithm>
#include <cmath>

namespace neuronix {

Matrix Softmax::forward(const Matrix& input) const {
    Matrix result = input;
    for (std::size_t c = 0; c < result.cols(); ++c) {
        double max_val = result(0, c);
        for (std::size_t r = 1; r < result.rows(); ++r)
            max_val = std::max(max_val, result(r, c));

        double sum = 0.0;
        for (std::size_t r = 0; r < result.rows(); ++r) {
            result(r, c) = std::exp(result(r, c) - max_val);
            sum += result(r, c);
        }
        for (std::size_t r = 0; r < result.rows(); ++r)
            result(r, c) /= sum;
    }
    return result;
}

} // namespace neuronix
