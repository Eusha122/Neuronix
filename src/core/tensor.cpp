#include "neuronix/core/tensor.hpp"
#include "rng_impl.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace neuronix {

// ─── Helpers ─────────────────────────────────────────────────────────────────

std::size_t Tensor::total_size(const Shape& shape) noexcept {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

void Tensor::init_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// ─── Constructors ─────────────────────────────────────────────────────────────

Tensor::Tensor(Shape shape)
    : shape_{std::move(shape)}, data_(total_size(shape_), 0.0) {
    init_strides();
}

Tensor::Tensor(Shape shape, double value)
    : shape_{std::move(shape)}, data_(total_size(shape_), value) {
    init_strides();
}

Tensor::Tensor(Shape shape, std::initializer_list<double> values)
    : shape_{std::move(shape)}, data_{values} {
    if (data_.size() != total_size(shape_)) {
        throw std::invalid_argument{
            "Tensor: initializer_list size does not match shape"};
    }
    init_strides();
}

Tensor::Tensor(Shape shape, std::vector<double> data)
    : shape_{std::move(shape)}, data_{std::move(data)} {
    if (data_.size() != total_size(shape_)) {
        throw std::invalid_argument{
            "Tensor: data vector size does not match shape"};
    }
    init_strides();
}

// ─── Static Factories ─────────────────────────────────────────────────────────

Tensor Tensor::zeros(Shape shape) {
    return Tensor{std::move(shape)};
}

Tensor Tensor::ones(Shape shape) {
    return Tensor{std::move(shape), 1.0};
}

Tensor Tensor::random_uniform(Shape shape, double low, double high) {
    if (low > high) {
        throw std::invalid_argument{"Tensor::random_uniform: low must be <= high"};
    }
    const std::size_t n = total_size(shape);
    std::vector<double> data(n);
    std::uniform_real_distribution<double> dist{low, high};
    for (auto& v : data) v = dist(detail::global_rng());
    return Tensor{std::move(shape), std::move(data)};
}

Tensor Tensor::random_normal(Shape shape, double mean, double stddev) {
    if (stddev < 0.0) {
        throw std::invalid_argument{"Tensor::random_normal: stddev must be >= 0"};
    }
    const std::size_t n = total_size(shape);
    std::vector<double> data(n);
    std::normal_distribution<double> dist{mean, stddev};
    for (auto& v : data) v = dist(detail::global_rng());
    return Tensor{std::move(shape), std::move(data)};
}

Tensor Tensor::from_matrix(const Matrix& m) {
    std::vector<double> data(m.data(), m.data() + m.size());
    return Tensor{Shape{m.rows(), m.cols()}, std::move(data)};
}

// ─── Properties ──────────────────────────────────────────────────────────────

std::size_t Tensor::dim(std::size_t axis) const {
    if (axis >= shape_.size()) {
        throw std::out_of_range{"Tensor::dim: axis out of range"};
    }
    return shape_[axis];
}

// ─── Bounds-checked element access ───────────────────────────────────────────

double& Tensor::at(std::initializer_list<std::size_t> indices) {
    if (indices.size() != rank()) {
        throw std::invalid_argument{"Tensor::at: wrong number of indices"};
    }
    auto it = indices.begin();
    for (std::size_t k = 0; k < rank(); ++k, ++it) {
        if (*it >= shape_[k]) {
            throw std::out_of_range{"Tensor::at: index out of range"};
        }
    }
    std::size_t flat = 0;
    it = indices.begin();
    for (std::size_t k = 0; k < rank(); ++k, ++it) {
        flat += *it * strides_[k];
    }
    return data_[flat];
}

const double& Tensor::at(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument{"Tensor::at: wrong number of indices"};
    }
    auto it = indices.begin();
    for (std::size_t k = 0; k < rank(); ++k, ++it) {
        if (*it >= shape_[k]) {
            throw std::out_of_range{"Tensor::at: index out of range"};
        }
    }
    std::size_t flat = 0;
    it = indices.begin();
    for (std::size_t k = 0; k < rank(); ++k, ++it) {
        flat += *it * strides_[k];
    }
    return data_[flat];
}

// ─── In-place arithmetic ──────────────────────────────────────────────────────

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument{"Tensor::operator+=: shape mismatch"};
    }
    for (std::size_t i = 0; i < data_.size(); ++i)
        data_[i] += other.data_[i];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument{"Tensor::operator-=: shape mismatch"};
    }
    for (std::size_t i = 0; i < data_.size(); ++i)
        data_[i] -= other.data_[i];
    return *this;
}

Tensor& Tensor::operator*=(double scalar) noexcept {
    for (auto& v : data_) v *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(double scalar) {
    if (scalar == 0.0) {
        throw std::invalid_argument{"Tensor::operator/=: division by zero"};
    }
    for (auto& v : data_) v /= scalar;
    return *this;
}

// ─── Binary arithmetic ────────────────────────────────────────────────────────

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result{*this};
    result += other;
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result{*this};
    result -= other;
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result{*this};
    result *= scalar;
    return result;
}

Tensor Tensor::operator/(double scalar) const {
    Tensor result{*this};
    result /= scalar;
    return result;
}

Tensor Tensor::operator-() const {
    Tensor result{*this};
    result *= -1.0;
    return result;
}

Tensor Tensor::hadamard(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument{"Tensor::hadamard: shape mismatch"};
    }
    Tensor result{shape_};
    for (std::size_t i = 0; i < data_.size(); ++i)
        result.data_[i] = data_[i] * other.data_[i];
    return result;
}

// ─── Shape operations ─────────────────────────────────────────────────────────

Tensor Tensor::reshape(Shape new_shape) const {
    if (total_size(new_shape) != data_.size()) {
        throw std::invalid_argument{
            "Tensor::reshape: total elements must be unchanged"};
    }
    Tensor result{std::move(new_shape), data_};
    return result;
}

Tensor Tensor::flatten() const {
    return reshape(Shape{data_.size()});
}

// ─── Reductions ───────────────────────────────────────────────────────────────

double Tensor::sum() const noexcept {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Tensor::mean() const {
    if (data_.empty()) throw std::logic_error{"Tensor::mean: empty tensor"};
    return sum() / static_cast<double>(data_.size());
}

double Tensor::min() const {
    if (data_.empty()) throw std::logic_error{"Tensor::min: empty tensor"};
    return *std::min_element(data_.begin(), data_.end());
}

double Tensor::max() const {
    if (data_.empty()) throw std::logic_error{"Tensor::max: empty tensor"};
    return *std::max_element(data_.begin(), data_.end());
}

double Tensor::frobenius_norm() const noexcept {
    double sq = 0.0;
    for (const auto& v : data_) sq += v * v;
    return std::sqrt(sq);
}

// ─── Apply ────────────────────────────────────────────────────────────────────

Tensor Tensor::apply(std::function<double(double)> func) const {
    Tensor result{shape_};
    for (std::size_t i = 0; i < data_.size(); ++i)
        result.data_[i] = func(data_[i]);
    return result;
}

// ─── Matrix interop ───────────────────────────────────────────────────────────

Matrix Tensor::to_matrix() const {
    if (rank() != 2) {
        throw std::invalid_argument{
            "Tensor::to_matrix: tensor must be rank 2"};
    }
    return Matrix{shape_[0], shape_[1], data_};
}

// ─── Comparison ───────────────────────────────────────────────────────────────

bool Tensor::operator==(const Tensor& other) const noexcept {
    return shape_ == other.shape_ && data_ == other.data_;
}

bool Tensor::operator!=(const Tensor& other) const noexcept {
    return !(*this == other);
}

bool Tensor::approx_equal(const Tensor& other, double tol) const noexcept {
    if (shape_ != other.shape_) return false;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (std::abs(data_[i] - other.data_[i]) > tol) return false;
    }
    return true;
}

// ─── Free functions ───────────────────────────────────────────────────────────

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}

} // namespace neuronix
