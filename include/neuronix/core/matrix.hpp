#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace neuronix {

class Matrix {
public:
    Matrix() noexcept : rows_{0}, cols_{0} {}
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, double value);
    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values);
    Matrix(std::size_t rows, std::size_t cols, std::vector<double> data);

    [[nodiscard]] static Matrix zeros(std::size_t rows, std::size_t cols);
    [[nodiscard]] static Matrix ones(std::size_t rows, std::size_t cols);
    [[nodiscard]] static Matrix identity(std::size_t n);
    [[nodiscard]] static Matrix random_uniform(std::size_t rows, std::size_t cols,
                                               double low = 0.0, double high = 1.0);
    [[nodiscard]] static Matrix random_normal(std::size_t rows, std::size_t cols,
                                              double mean = 0.0, double stddev = 1.0);
    [[nodiscard]] static Matrix xavier_uniform(std::size_t rows, std::size_t cols);
    [[nodiscard]] static Matrix xavier_normal(std::size_t rows, std::size_t cols);
    [[nodiscard]] static Matrix he_uniform(std::size_t rows, std::size_t cols);
    [[nodiscard]] static Matrix he_normal(std::size_t rows, std::size_t cols);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    [[nodiscard]] double& operator()(std::size_t row, std::size_t col) noexcept {
        return data_[row * cols_ + col];
    }
    [[nodiscard]] const double& operator()(std::size_t row, std::size_t col) const noexcept {
        return data_[row * cols_ + col];
    }

    [[nodiscard]] double& at(std::size_t row, std::size_t col);
    [[nodiscard]] const double& at(std::size_t row, std::size_t col) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar) noexcept;
    Matrix& operator/=(double scalar);

    [[nodiscard]] Matrix operator+(const Matrix& other) const;
    [[nodiscard]] Matrix operator-(const Matrix& other) const;
    [[nodiscard]] Matrix operator*(double scalar) const;
    [[nodiscard]] Matrix operator/(double scalar) const;
    [[nodiscard]] Matrix operator-() const;

    // Matrix multiplication (not element-wise)
    [[nodiscard]] Matrix operator*(const Matrix& other) const;

    [[nodiscard]] Matrix hadamard(const Matrix& other) const;
    [[nodiscard]] Matrix transpose() const;

    [[nodiscard]] double sum() const noexcept;
    [[nodiscard]] double mean() const;
    [[nodiscard]] double min() const;
    [[nodiscard]] double max() const;
    [[nodiscard]] double frobenius_norm() const noexcept;

    [[nodiscard]] Matrix apply(std::function<double(double)> func) const;

    [[nodiscard]] bool operator==(const Matrix& other) const noexcept;
    [[nodiscard]] bool operator!=(const Matrix& other) const noexcept;
    [[nodiscard]] bool approx_equal(const Matrix& other, double tol = 1e-9) const noexcept;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

[[nodiscard]] Matrix operator*(double scalar, const Matrix& m);

void seed_random(std::uint64_t seed);

} // namespace neuronix
