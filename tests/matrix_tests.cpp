#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/core/matrix.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ─── Construction ───────────────────────────────────────────────────────────

TEST_CASE("Matrix zero-initialized construction", "[matrix][construction]") {
    Matrix m(2, 3);
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    REQUIRE(m.size() == 6);
    REQUIRE_FALSE(m.empty());
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            REQUIRE(m(i, j) == 0.0);
}

TEST_CASE("Matrix fill-value construction", "[matrix][construction]") {
    Matrix m(3, 2, 7.5);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            REQUIRE(m(i, j) == 7.5);
}

TEST_CASE("Matrix initializer_list construction", "[matrix][construction]") {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(0, 1) == 2.0);
    REQUIRE(m(1, 0) == 3.0);
    REQUIRE(m(1, 1) == 4.0);
}

TEST_CASE("Matrix vector construction", "[matrix][construction]") {
    std::vector<double> data{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    Matrix m(2, 3, data);
    REQUIRE(m(0, 0) == 10.0);
    REQUIRE(m(0, 1) == 20.0);
    REQUIRE(m(1, 2) == 60.0);
}

TEST_CASE("Matrix empty construction", "[matrix][construction]") {
    Matrix m(0, 0);
    REQUIRE(m.empty());
    REQUIRE(m.size() == 0);
    REQUIRE(m.rows() == 0);
    REQUIRE(m.cols() == 0);
}

TEST_CASE("Matrix initializer_list size mismatch throws", "[matrix][construction]") {
    REQUIRE_THROWS_AS((Matrix(2, 2, {1.0, 2.0, 3.0})), std::invalid_argument);
}

TEST_CASE("Matrix vector size mismatch throws", "[matrix][construction]") {
    std::vector<double> bad{1.0, 2.0};
    REQUIRE_THROWS_AS((Matrix(2, 2, std::move(bad))), std::invalid_argument);
}

// ─── Static Factories ────────────────────────────────────────────────────────

TEST_CASE("Matrix::zeros", "[matrix][factories]") {
    auto m = Matrix::zeros(3, 4);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 4);
    REQUIRE(m.sum() == 0.0);
}

TEST_CASE("Matrix::ones", "[matrix][factories]") {
    auto m = Matrix::ones(2, 5);
    REQUIRE(m.sum() == 10.0);
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(1, 4) == 1.0);
}

TEST_CASE("Matrix::identity", "[matrix][factories]") {
    auto m = Matrix::identity(3);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 3);
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(1, 1) == 1.0);
    REQUIRE(m(2, 2) == 1.0);
    REQUIRE(m(0, 1) == 0.0);
    REQUIRE(m(0, 2) == 0.0);
    REQUIRE(m(1, 0) == 0.0);
    REQUIRE(m(1, 2) == 0.0);
    REQUIRE(m(2, 0) == 0.0);
    REQUIRE(m(2, 1) == 0.0);
}

TEST_CASE("Matrix::random_uniform range", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::random_uniform(50, 50, -1.0, 1.0);
    REQUIRE(m.rows() == 50);
    REQUIRE(m.cols() == 50);
    REQUIRE(m.min() >= -1.0);
    REQUIRE(m.max() <= 1.0);
}

TEST_CASE("Matrix::random_uniform low > high throws", "[matrix][factories]") {
    REQUIRE_THROWS_AS(Matrix::random_uniform(2, 2, 1.0, 0.0), std::invalid_argument);
}

TEST_CASE("Matrix::random_normal shape", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::random_normal(10, 10, 0.0, 1.0);
    REQUIRE(m.rows() == 10);
    REQUIRE(m.cols() == 10);
    REQUIRE(m.size() == 100);
}

TEST_CASE("Matrix::random_normal negative stddev throws", "[matrix][factories]") {
    REQUIRE_THROWS_AS(Matrix::random_normal(2, 2, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("Matrix::xavier_uniform shape and range", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::xavier_uniform(10, 20);
    REQUIRE(m.rows() == 10);
    REQUIRE(m.cols() == 20);
    // Xavier uniform limit = sqrt(6 / (10+20)) ≈ 0.447
    const double limit = std::sqrt(6.0 / 30.0);
    REQUIRE(m.min() >= -limit - 1e-12);
    REQUIRE(m.max() <=  limit + 1e-12);
}

TEST_CASE("Matrix::xavier_normal shape", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::xavier_normal(10, 20);
    REQUIRE(m.rows() == 10);
    REQUIRE(m.cols() == 20);
}

TEST_CASE("Matrix::he_uniform shape", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::he_uniform(10, 20);
    REQUIRE(m.rows() == 10);
    REQUIRE(m.cols() == 20);
}

TEST_CASE("Matrix::he_normal shape", "[matrix][factories]") {
    seed_random(42);
    auto m = Matrix::he_normal(10, 20);
    REQUIRE(m.rows() == 10);
    REQUIRE(m.cols() == 20);
}

// ─── Element Access ──────────────────────────────────────────────────────────

TEST_CASE("Matrix operator() read and write", "[matrix][access]") {
    Matrix m(2, 3);
    m(0, 2) = 5.5;
    m(1, 0) = -3.0;
    REQUIRE(m(0, 2) == 5.5);
    REQUIRE(m(1, 0) == -3.0);
    REQUIRE(m(0, 0) == 0.0);
}

TEST_CASE("Matrix at() read and write", "[matrix][access]") {
    Matrix m(2, 3, 1.0);
    m.at(1, 1) = 99.0;
    REQUIRE(m.at(1, 1) == 99.0);
    REQUIRE(m.at(0, 0) == 1.0);
}

TEST_CASE("Matrix at() row out of range throws", "[matrix][access]") {
    Matrix m(2, 3);
    REQUIRE_THROWS_AS(m.at(2, 0), std::out_of_range);
}

TEST_CASE("Matrix at() col out of range throws", "[matrix][access]") {
    Matrix m(2, 3);
    REQUIRE_THROWS_AS(m.at(0, 3), std::out_of_range);
}

TEST_CASE("Matrix data() pointer is row-major", "[matrix][access]") {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(m.data()[0] == 1.0);
    REQUIRE(m.data()[1] == 2.0);
    REQUIRE(m.data()[2] == 3.0);
    REQUIRE(m.data()[3] == 4.0);
}

// ─── Element-wise Arithmetic ─────────────────────────────────────────────────

TEST_CASE("Matrix operator+", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    auto c = a + b;
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(0, 1) == 8.0);
    REQUIRE(c(1, 0) == 10.0);
    REQUIRE(c(1, 1) == 12.0);
    REQUIRE(a(0, 0) == 1.0); // original unchanged
}

TEST_CASE("Matrix operator-", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    auto c = b - a;
    REQUIRE(c(0, 0) == 4.0);
    REQUIRE(c(1, 1) == 4.0);
}

TEST_CASE("Matrix operator+=", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    a += b;
    REQUIRE(a(0, 0) == 6.0);
    REQUIRE(a(1, 1) == 12.0);
}

TEST_CASE("Matrix operator-=", "[matrix][arithmetic]") {
    Matrix a(2, 2, {5.0, 6.0, 7.0, 8.0});
    Matrix b(2, 2, {1.0, 2.0, 3.0, 4.0});
    a -= b;
    REQUIRE(a(0, 0) == 4.0);
    REQUIRE(a(1, 1) == 4.0);
}

TEST_CASE("Matrix scalar multiply member", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    auto c = a * 2.0;
    REQUIRE(c(0, 0) == 2.0);
    REQUIRE(c(0, 1) == 4.0);
    REQUIRE(c(1, 0) == 6.0);
    REQUIRE(c(1, 1) == 8.0);
}

TEST_CASE("Matrix scalar multiply free function", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    auto c = 3.0 * a;
    REQUIRE(c(0, 0) == 3.0);
    REQUIRE(c(1, 1) == 12.0);
}

TEST_CASE("Matrix scalar divide", "[matrix][arithmetic]") {
    Matrix a(2, 2, {2.0, 4.0, 6.0, 8.0});
    auto c = a / 2.0;
    REQUIRE(c(0, 0) == 1.0);
    REQUIRE(c(1, 1) == 4.0);
}

TEST_CASE("Matrix divide by zero throws", "[matrix][arithmetic]") {
    Matrix a(2, 2, 1.0);
    REQUIRE_THROWS_AS(a / 0.0, std::invalid_argument);
    REQUIRE_THROWS_AS(a /= 0.0, std::invalid_argument);
}

TEST_CASE("Matrix unary negation", "[matrix][arithmetic]") {
    Matrix a(2, 2, {1.0, -2.0, 3.0, -4.0});
    auto c = -a;
    REQUIRE(c(0, 0) == -1.0);
    REQUIRE(c(0, 1) == 2.0);
    REQUIRE(c(1, 0) == -3.0);
    REQUIRE(c(1, 1) == 4.0);
    REQUIRE(a(0, 0) == 1.0); // original unchanged
}

TEST_CASE("Matrix element-wise dimension mismatch throws", "[matrix][arithmetic]") {
    Matrix a(2, 2);
    Matrix c(3, 2);
    REQUIRE_THROWS_AS(a + c,  std::invalid_argument);
    REQUIRE_THROWS_AS(a - c,  std::invalid_argument);
    REQUIRE_THROWS_AS(a += c, std::invalid_argument);
    REQUIRE_THROWS_AS(a -= c, std::invalid_argument);
}

// ─── Matrix Multiplication ───────────────────────────────────────────────────

TEST_CASE("Matrix multiply 2x3 * 3x2 = 2x2", "[matrix][matmul]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0});
    Matrix b(3, 2, {7.0,  8.0,
                    9.0,  10.0,
                    11.0, 12.0});
    auto c = a * b;
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
    REQUIRE(c(0, 0) == 58.0);
    REQUIRE(c(0, 1) == 64.0);
    REQUIRE(c(1, 0) == 139.0);
    REQUIRE(c(1, 1) == 154.0);
}

TEST_CASE("Matrix multiply identity leaves matrix unchanged", "[matrix][matmul]") {
    Matrix a(3, 3, {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0});
    auto id = Matrix::identity(3);
    REQUIRE((id * a) == a);
    REQUIRE((a * id) == a);
}

TEST_CASE("Matrix multiply 1x1", "[matrix][matmul]") {
    Matrix a(1, 1, {5.0});
    Matrix b(1, 1, {3.0});
    auto c = a * b;
    REQUIRE(c.rows() == 1);
    REQUIRE(c.cols() == 1);
    REQUIRE(c(0, 0) == 15.0);
}

TEST_CASE("Matrix multiply dimension mismatch throws", "[matrix][matmul]") {
    Matrix a(2, 3);
    Matrix b(2, 2);
    REQUIRE_THROWS_AS(a * b, std::invalid_argument);
}

// ─── Hadamard Product ────────────────────────────────────────────────────────

TEST_CASE("Matrix hadamard product", "[matrix][hadamard]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    auto c = a.hadamard(b);
    REQUIRE(c(0, 0) == 5.0);
    REQUIRE(c(0, 1) == 12.0);
    REQUIRE(c(1, 0) == 21.0);
    REQUIRE(c(1, 1) == 32.0);
}

TEST_CASE("Matrix hadamard dimension mismatch throws", "[matrix][hadamard]") {
    Matrix a(2, 2);
    Matrix b(2, 3);
    REQUIRE_THROWS_AS(a.hadamard(b), std::invalid_argument);
}

// ─── Transpose ───────────────────────────────────────────────────────────────

TEST_CASE("Matrix transpose 2x3 -> 3x2", "[matrix][transpose]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0});
    auto t = a.transpose();
    REQUIRE(t.rows() == 3);
    REQUIRE(t.cols() == 2);
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(0, 1) == 4.0);
    REQUIRE(t(1, 0) == 2.0);
    REQUIRE(t(1, 1) == 5.0);
    REQUIRE(t(2, 0) == 3.0);
    REQUIRE(t(2, 1) == 6.0);
}

TEST_CASE("Matrix transpose is its own inverse", "[matrix][transpose]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(a.transpose().transpose() == a);
}

TEST_CASE("Matrix square transpose", "[matrix][transpose]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    auto t = a.transpose();
    REQUIRE(t(0, 1) == 3.0);
    REQUIRE(t(1, 0) == 2.0);
}

// ─── Reductions ──────────────────────────────────────────────────────────────

TEST_CASE("Matrix sum", "[matrix][reductions]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(a.sum() == 21.0);
}

TEST_CASE("Matrix mean", "[matrix][reductions]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(a.mean() == 3.5);
}

TEST_CASE("Matrix min", "[matrix][reductions]") {
    Matrix a(2, 3, {3.0, 1.0, 4.0, 1.0, 5.0, 9.0});
    REQUIRE(a.min() == 1.0);
}

TEST_CASE("Matrix max", "[matrix][reductions]") {
    Matrix a(2, 3, {3.0, 1.0, 4.0, 1.0, 5.0, 9.0});
    REQUIRE(a.max() == 9.0);
}

TEST_CASE("Matrix frobenius_norm", "[matrix][reductions]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91)
    REQUIRE_THAT(a.frobenius_norm(), WithinAbs(std::sqrt(91.0), 1e-9));
}

TEST_CASE("Matrix sum and frobenius_norm on empty matrix", "[matrix][reductions]") {
    Matrix empty(0, 0);
    REQUIRE(empty.sum() == 0.0);
    REQUIRE(empty.frobenius_norm() == 0.0);
}

TEST_CASE("Matrix mean on empty matrix throws", "[matrix][reductions]") {
    REQUIRE_THROWS_AS(Matrix(0, 0).mean(), std::logic_error);
}

TEST_CASE("Matrix min on empty matrix throws", "[matrix][reductions]") {
    REQUIRE_THROWS_AS(Matrix(0, 0).min(), std::logic_error);
}

TEST_CASE("Matrix max on empty matrix throws", "[matrix][reductions]") {
    REQUIRE_THROWS_AS(Matrix(0, 0).max(), std::logic_error);
}

// ─── Apply ───────────────────────────────────────────────────────────────────

TEST_CASE("Matrix apply identity", "[matrix][apply]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    auto b = a.apply([](double x) { return x; });
    REQUIRE(b == a);
}

TEST_CASE("Matrix apply ReLU", "[matrix][apply]") {
    Matrix a(2, 2, {-2.0, -1.0, 0.0, 1.0});
    auto b = a.apply([](double x) { return x > 0.0 ? x : 0.0; });
    REQUIRE(b(0, 0) == 0.0);
    REQUIRE(b(0, 1) == 0.0);
    REQUIRE(b(1, 0) == 0.0);
    REQUIRE(b(1, 1) == 1.0);
}

TEST_CASE("Matrix apply square", "[matrix][apply]") {
    Matrix a(1, 4, {-2.0, -1.0, 0.0, 3.0});
    auto b = a.apply([](double x) { return x * x; });
    REQUIRE(b(0, 0) == 4.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(0, 2) == 0.0);
    REQUIRE(b(0, 3) == 9.0);
}

TEST_CASE("Matrix apply does not modify original", "[matrix][apply]") {
    Matrix a(1, 2, {1.0, 2.0});
    (void)a.apply([](double x) { return x * 100.0; });
    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 2.0);
}

// ─── Comparison ──────────────────────────────────────────────────────────────

TEST_CASE("Matrix operator== equal matrices", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(a == b);
}

TEST_CASE("Matrix operator!= different values", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix c(2, 2, {1.0, 2.0, 3.0, 5.0});
    REQUIRE(a != c);
}

TEST_CASE("Matrix operator!= different dimensions", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix d(1, 4, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(a != d);
}

TEST_CASE("Matrix approx_equal within tolerance", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {1.0 + 1e-10, 2.0, 3.0, 4.0});
    REQUIRE(a.approx_equal(b));
}

TEST_CASE("Matrix approx_equal outside tolerance", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {1.0 + 1e-8, 2.0, 3.0, 4.0});
    REQUIRE_FALSE(a.approx_equal(b));
}

TEST_CASE("Matrix approx_equal different dimensions", "[matrix][comparison]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(1, 4, {1.0, 2.0, 3.0, 4.0});
    REQUIRE_FALSE(a.approx_equal(b));
}

// ─── Mathematical Properties ─────────────────────────────────────────────────

TEST_CASE("Matrix A * I = A", "[matrix][properties]") {
    Matrix a(3, 3, {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0});
    auto id = Matrix::identity(3);
    REQUIRE((a * id).approx_equal(a));
    REQUIRE((id * a).approx_equal(a));
}

TEST_CASE("Matrix (A + B)^T = A^T + B^T", "[matrix][properties]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix b(2, 3, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    REQUIRE((a + b).transpose().approx_equal(a.transpose() + b.transpose()));
}

TEST_CASE("Matrix (AB)^T = B^T * A^T", "[matrix][properties]") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix b(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto lhs = (a * b).transpose();
    auto rhs = b.transpose() * a.transpose();
    REQUIRE(lhs.approx_equal(rhs));
}

TEST_CASE("Matrix scalar distributes over addition: k*(A+B) = kA + kB", "[matrix][properties]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});

    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0}) ;
    REQUIRE((2.0 * (a + b)).approx_equal(2.0 * a + 2.0 * b));
}

TEST_CASE("Matrix hadamard is commutative", "[matrix][properties]") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {5.0, 6.0, 7.0, 8.0});
    
    REQUIRE(a.hadamard(b) == b.hadamard(a));
}

TEST_CASE("Matrix seeded random is reproducible", "[matrix][random]") {
    seed_random(99);

    auto m1 = Matrix::random_uniform(5, 5);

    seed_random(99);

    auto m2 = Matrix::random_uniform(5, 5);
    REQUIRE(m1 == m2);
}
