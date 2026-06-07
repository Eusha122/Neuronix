#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <vector>

#include "neuronix/core/matrix.hpp"

namespace neuronix {

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    explicit Tensor(Shape shape);
    Tensor(Shape shape, double value);
    Tensor(Shape shape, std::initializer_list<double> values);
    Tensor(Shape shape, std::vector<double> data);

    [[nodiscard]] static Tensor zeros(Shape shape);
    [[nodiscard]] static Tensor ones(Shape shape);
    [[nodiscard]] static Tensor random_uniform(Shape shape, double low = 0.0, double high = 1.0);
    [[nodiscard]] static Tensor random_normal(Shape shape, double mean = 0.0, double stddev = 1.0);
    [[nodiscard]] static Tensor from_matrix(const Matrix& m);

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] std::size_t dim(std::size_t axis) const;
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    // Flat (linear) index access
    [[nodiscard]] double& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const double& operator[](std::size_t i) const noexcept { return data_[i]; }

    // Multi-dimensional unchecked access: t(i, j, k, ...)
    template<typename... Idx>
    [[nodiscard]] double& operator()(Idx... indices) noexcept {
        const std::size_t arr[] = {static_cast<std::size_t>(indices)...};
        std::size_t flat = 0;
        for (std::size_t k = 0; k < sizeof...(indices); ++k)
            flat += arr[k] * strides_[k];
        return data_[flat];
    }

    template<typename... Idx>
    [[nodiscard]] const double& operator()(Idx... indices) const noexcept {
        const std::size_t arr[] = {static_cast<std::size_t>(indices)...};
        std::size_t flat = 0;
        for (std::size_t k = 0; k < sizeof...(indices); ++k)
            flat += arr[k] * strides_[k];
        return data_[flat];
    }

    // Bounds-checked access: t.at({i, j, k})
    [[nodiscard]] double& at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const double& at(std::initializer_list<std::size_t> indices) const;

    // In-place arithmetic
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(double scalar) noexcept;
    Tensor& operator/=(double scalar);

    // Binary arithmetic
    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(double scalar) const;
    [[nodiscard]] Tensor operator/(double scalar) const;
    [[nodiscard]] Tensor operator-() const;

    // Element-wise (Hadamard) product
    [[nodiscard]] Tensor hadamard(const Tensor& other) const;

    // Shape operations
    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    [[nodiscard]] Tensor flatten() const;

    // Reductions
    [[nodiscard]] double sum() const noexcept;
    [[nodiscard]] double mean() const;
    [[nodiscard]] double min() const;
    [[nodiscard]] double max() const;
    [[nodiscard]] double frobenius_norm() const noexcept;

    // Element-wise function application
    [[nodiscard]] Tensor apply(std::function<double(double)> func) const;

    // Matrix interop (rank-2 tensors only)
    [[nodiscard]] Matrix to_matrix() const;

    // Comparison
    [[nodiscard]] bool operator==(const Tensor& other) const noexcept;
    [[nodiscard]] bool operator!=(const Tensor& other) const noexcept;
    [[nodiscard]] bool approx_equal(const Tensor& other, double tol = 1e-9) const noexcept;

private:
    Shape shape_;
    std::vector<std::size_t> strides_;
    std::vector<double> data_;

    void init_strides();
    static std::size_t total_size(const Shape& shape) noexcept;
};

[[nodiscard]] Tensor operator*(double scalar, const Tensor& t);

} // namespace neuronix
