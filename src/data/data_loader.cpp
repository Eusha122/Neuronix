#include "neuronix/data/data_loader.hpp"
#include "../core/rng_impl.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace neuronix {

DataLoader::DataLoader(const Matrix& X, const Matrix& y,
                       std::size_t batch_size, bool shuffle)
    : X_{X}, y_{y}, batch_size_{batch_size}, shuffle_{shuffle}, n_{X.cols()} {
    if (X.cols() != y.cols())
        throw std::invalid_argument{"DataLoader: X and y column counts must match"};
    if (batch_size == 0)
        throw std::invalid_argument{"DataLoader: batch_size must be > 0"};

    indices_.resize(n_);
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_)
        std::shuffle(indices_.begin(), indices_.end(), detail::global_rng());
}

void DataLoader::reshuffle() {
    if (shuffle_)
        std::shuffle(indices_.begin(), indices_.end(), detail::global_rng());
}

std::size_t DataLoader::num_batches() const noexcept {
    return n_ / batch_size_;   // drop incomplete last batch
}

DataLoader::Batch DataLoader::make_batch(std::size_t batch_idx) const {
    const std::size_t start = batch_idx * batch_size_;
    Matrix Xb = Matrix::zeros(X_.rows(), batch_size_);
    Matrix yb = Matrix::zeros(y_.rows(), batch_size_);
    for (std::size_t i = 0; i < batch_size_; ++i) {
        const std::size_t col = indices_[start + i];
        for (std::size_t r = 0; r < X_.rows(); ++r) Xb(r, i) = X_(r, col);
        for (std::size_t r = 0; r < y_.rows(); ++r) yb(r, i) = y_(r, col);
    }
    return {std::move(Xb), std::move(yb)};
}

DataLoader::Batch DataLoader::iterator::operator*() const {
    return dl_.make_batch(batch_idx_);
}

} // namespace neuronix
