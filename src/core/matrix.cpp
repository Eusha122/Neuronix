#include "neuronix/core/matrix.hpp"
#include "rng_impl.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace neuronix {

// Constructors

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_{rows}, cols_{cols}, data_(rows * cols, 0.0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_{rows}, cols_{cols}, data_(rows * cols, value) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values)
    : rows_{rows}, cols_{cols}, data_{values} {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument{
            "Matrix: initializer_list size does not match rows * cols"};
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<double> data)
    : rows_{rows}, cols_{cols}, data_{std::move(data)} {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument{
            "Matrix: data vector size does not match rows * cols"};
    }
}

// Static factories

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix{rows, cols};
}

Matrix Matrix::ones(std::size_t rows, std::size_t cols) {
    return Matrix{rows, cols, 1.0};
}

Matrix Matrix::identity(std::size_t n) {
    Matrix m{n, n};
    for (std::size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::random_uniform(std::size_t rows, std::size_t cols, double low, double high) {
    if (low > high) {
        throw std::invalid_argument{"Matrix::random_uniform: low must be <= high"};
    }
    Matrix m{rows, cols};
    std::uniform_real_distribution<double> dist{low, high};
    for (auto& v : m.data_) {
        v = dist(detail::global_rng());
    }
    return m;
}

Matrix Matrix::random_normal(std::size_t rows, std::size_t cols, double mean, double stddev) {
    if (stddev < 0.0) {
        throw std::invalid_argument{"Matrix::random_normal: stddev must be >= 0"};
    }
    Matrix m{rows, cols};
    std::normal_distribution<double> dist{mean, stddev};
    for (auto& v : m.data_) {
        v = dist(detail::global_rng());
    }
    return m;
}

Matrix Matrix::xavier_uniform(std::size_t rows, std::size_t cols) {
    const double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
    return random_uniform(rows, cols, -limit, limit);
}

Matrix Matrix::xavier_normal(std::size_t rows, std::size_t cols) {
    const double stddev = std::sqrt(2.0 / static_cast<double>(rows + cols));
    return random_normal(rows, cols, 0.0, stddev);
}

Matrix Matrix::he_uniform(std::size_t rows, std::size_t cols) {
    const double limit = std::sqrt(6.0 / static_cast<double>(rows));
    return random_uniform(rows, cols, -limit, limit);
}

Matrix Matrix::he_normal(std::size_t rows, std::size_t cols) {
    const double stddev = std::sqrt(2.0 / static_cast<double>(rows));
    return random_normal(rows, cols, 0.0, stddev);
}

// Bounds-checked element access

double& Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{"Matrix::at: index out of range"};
    }
    return data_[row * cols_ + col];
}

const double& Matrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{"Matrix::at: index out of range"};
    }
    return data_[row * cols_ + col];
}

// In-place arithmetic

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument{"Matrix::operator+=: dimension mismatch"};
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument{"Matrix::operator-=: dimension mismatch"};
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) noexcept {
    for (auto& v : data_) {
        v *= scalar;
    }
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (scalar == 0.0) {
        throw std::invalid_argument{"Matrix::operator/=: division by zero"};
    }
    for (auto& v : data_) {
        v /= scalar;
    }
    return *this;
}

// Binary arithmetic

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result{*this};
    result += other;
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result{*this};
    result -= other;
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result{*this};
    result *= scalar;
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result{*this};
    result /= scalar;
    return result;
}

Matrix Matrix::operator-() const {
    Matrix result{*this};
    result *= -1.0;
    return result;
}

// Matrix multiplication — ikj loop order, parallelised over output rows.
// Each thread owns distinct rows of result so no data race.
// ptrdiff_t loop variable required by OpenMP 2.0 (MSVC default).

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument{
            "Matrix::operator*: left cols must equal right rows"};
    }
    Matrix result{rows_, other.cols_};
    const auto   M = static_cast<std::ptrdiff_t>(rows_);
    const auto   K = static_cast<std::ptrdiff_t>(cols_);
    const auto   N = static_cast<std::ptrdiff_t>(other.cols_);
    const double* A = data_.data();
    const double* B = other.data_.data();
    double*       C = result.data_.data();

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < M; ++i) {
        for (std::ptrdiff_t k = 0; k < K; ++k) {
            const double aik = A[i * K + k];
            for (std::ptrdiff_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
    return result;
}

// Hadamard (element-wise) product

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument{"Matrix::hadamard: dimension mismatch"};
    }
    Matrix result{rows_, cols_};
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Transpose

Matrix Matrix::transpose() const {
    Matrix result{cols_, rows_};
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Reductions

double Matrix::sum() const noexcept {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Matrix::mean() const {
    if (data_.empty()) {
        throw std::logic_error{"Matrix::mean: called on empty matrix"};
    }
    return sum() / static_cast<double>(data_.size());
}

double Matrix::min() const {
    if (data_.empty()) {
        throw std::logic_error{"Matrix::min: called on empty matrix"};
    }
    return *std::min_element(data_.begin(), data_.end());
}

double Matrix::max() const {
    if (data_.empty()) {
        throw std::logic_error{"Matrix::max: called on empty matrix"};
    }
    return *std::max_element(data_.begin(), data_.end());
}

double Matrix::frobenius_norm() const noexcept {
    double sq_sum = 0.0;
    for (const auto& v : data_) {
        sq_sum += v * v;
    }
    return std::sqrt(sq_sum);
}

// Element-wise application

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result{rows_, cols_};
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = func(data_[i]);
    }
    return result;
}

// Comparison

bool Matrix::operator==(const Matrix& other) const noexcept {
    return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
}

bool Matrix::operator!=(const Matrix& other) const noexcept {
    return !(*this == other);
}

bool Matrix::approx_equal(const Matrix& other, double tol) const noexcept {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (std::abs(data_[i] - other.data_[i]) > tol) return false;
    }
    return true;
}

// Free functions

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

void seed_random(std::uint64_t seed) {
    detail::global_rng().seed(seed);
}

} // namespace neuronix
