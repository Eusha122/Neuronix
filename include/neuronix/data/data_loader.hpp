#pragma once

#include <cstddef>
#include <vector>

#include "neuronix/core/matrix.hpp"

namespace neuronix {

// Mini-batch iterator over a (features x N) dataset.
// Holds const references — the dataset matrices must outlive the DataLoader.
class DataLoader {
public:
    struct Batch {
        Matrix X;  // (features, batch_size)
        Matrix y;  // (classes,  batch_size)
    };

    // shuffle=true randomly permutes samples before each epoch.
    // Incomplete final batch is dropped.
    DataLoader(const Matrix& X, const Matrix& y,
               std::size_t batch_size, bool shuffle = true);

    // Reshuffle indices for a new epoch (call at the start of each epoch).
    void reshuffle();

    [[nodiscard]] std::size_t num_batches()  const noexcept;
    [[nodiscard]] std::size_t batch_size()   const noexcept { return batch_size_; }
    [[nodiscard]] std::size_t num_samples()  const noexcept { return n_; }

    // Range-for support
    class iterator {
    public:
        iterator(const DataLoader& dl, std::size_t batch_idx)
            : dl_{dl}, batch_idx_{batch_idx} {}
        [[nodiscard]] Batch operator*()  const;
        iterator& operator++()           { ++batch_idx_; return *this; }
        bool operator!=(const iterator& o) const { return batch_idx_ != o.batch_idx_; }
    private:
        const DataLoader& dl_;
        std::size_t       batch_idx_;
    };

    [[nodiscard]] iterator begin() const { return {*this, 0}; }
    [[nodiscard]] iterator end()   const { return {*this, num_batches()}; }

private:
    const Matrix&              X_;
    const Matrix&              y_;
    std::size_t                batch_size_;
    bool                       shuffle_;
    std::size_t                n_;          // total samples
    std::vector<std::size_t>   indices_;    // permuted sample indices

    [[nodiscard]] Batch make_batch(std::size_t batch_idx) const;
};

} // namespace neuronix
