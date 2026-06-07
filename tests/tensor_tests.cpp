#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/core/tensor.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ─── Construction ────────────────────────────────────────────────────────────

TEST_CASE("Tensor zero-initialized construction", "[tensor][construction]") {
    Tensor t(Tensor::Shape{2, 3});
    REQUIRE(t.rank() == 2);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.size() == 6);
    REQUIRE_FALSE(t.empty());
    for (std::size_t i = 0; i < t.size(); ++i)
        REQUIRE(t[i] == 0.0);
}

TEST_CASE("Tensor 3D construction", "[tensor][construction]") {
    Tensor t(Tensor::Shape{2, 3, 4});
    REQUIRE(t.rank() == 3);
    REQUIRE(t.size() == 24);
}

TEST_CASE("Tensor 1D construction", "[tensor][construction]") {
    Tensor t(Tensor::Shape{6});
    REQUIRE(t.rank() == 1);
    REQUIRE(t.size() == 6);
}

TEST_CASE("Tensor fill-value construction", "[tensor][construction]") {
    Tensor t(Tensor::Shape{2, 3}, 5.0);
    for (std::size_t i = 0; i < t.size(); ++i)
        REQUIRE(t[i] == 5.0);
}

TEST_CASE("Tensor initializer_list construction", "[tensor][construction]") {
    Tensor t(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(t[0] == 1.0);
    REQUIRE(t[1] == 2.0);
    REQUIRE(t[2] == 3.0);
    REQUIRE(t[3] == 4.0);
}

TEST_CASE("Tensor vector construction", "[tensor][construction]") {
    std::vector<double> data{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    Tensor t(Tensor::Shape{2, 3}, data);
    REQUIRE(t[0] == 10.0);
    REQUIRE(t[5] == 60.0);
}

TEST_CASE("Tensor initializer_list size mismatch throws", "[tensor][construction]") {
    REQUIRE_THROWS_AS((Tensor(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0})), std::invalid_argument);
}

TEST_CASE("Tensor vector size mismatch throws", "[tensor][construction]") {
    std::vector<double> bad{1.0, 2.0};
    REQUIRE_THROWS_AS((Tensor(Tensor::Shape{2, 2}, std::move(bad))), std::invalid_argument);
}

// ─── Shape and properties ────────────────────────────────────────────────────

TEST_CASE("Tensor dim(axis)", "[tensor][shape]") {
    Tensor t(Tensor::Shape{2, 3, 4});
    REQUIRE(t.dim(0) == 2);
    REQUIRE(t.dim(1) == 3);
    REQUIRE(t.dim(2) == 4);
}

TEST_CASE("Tensor dim() out of range throws", "[tensor][shape]") {
    Tensor t(Tensor::Shape{2, 3});
    REQUIRE_THROWS_AS(t.dim(2), std::out_of_range);
}

TEST_CASE("Tensor empty", "[tensor][shape]") {
    Tensor t(Tensor::Shape{0, 3});
    REQUIRE(t.empty());
    REQUIRE(t.size() == 0);
}

// ─── Strides and element access ──────────────────────────────────────────────

TEST_CASE("Tensor 2D operator() matches flat index", "[tensor][access]") {
    // shape {2, 3}: strides {3, 1}
    // (1, 2) -> 1*3 + 2 = 5
    Tensor t(Tensor::Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(0, 1) == 2.0);
    REQUIRE(t(0, 2) == 3.0);
    REQUIRE(t(1, 0) == 4.0);
    REQUIRE(t(1, 1) == 5.0);
    REQUIRE(t(1, 2) == 6.0);
}

TEST_CASE("Tensor 3D operator() strides", "[tensor][access]") {
    // shape {2, 3, 4}: strides {12, 4, 1}
    // (1, 2, 3) -> 1*12 + 2*4 + 3*1 = 23
    Tensor t(Tensor::Shape{2, 3, 4});
    t(0, 0, 0) = 7.0;
    t(1, 2, 3) = 99.0;
    REQUIRE(t[0]  == 7.0);
    REQUIRE(t[23] == 99.0);
}

TEST_CASE("Tensor 1D operator() flat access", "[tensor][access]") {
    Tensor t(Tensor::Shape{5}, {10.0, 20.0, 30.0, 40.0, 50.0});
    REQUIRE(t(0) == 10.0);
    REQUIRE(t(4) == 50.0);
}

TEST_CASE("Tensor operator() write", "[tensor][access]") {
    Tensor t(Tensor::Shape{2, 2});
    t(0, 1) = 3.14;
    REQUIRE(t(0, 1) == 3.14);
    REQUIRE(t[1] == 3.14);
}

TEST_CASE("Tensor at() valid access", "[tensor][access]") {
    Tensor t(Tensor::Shape{2, 3}, 1.0);
    t.at({1, 2}) = 42.0;
    REQUIRE(t.at({1, 2}) == 42.0);
    REQUIRE(t.at({0, 0}) == 1.0);
}

TEST_CASE("Tensor at() index out of range throws", "[tensor][access]") {
    Tensor t(Tensor::Shape{2, 3});
    REQUIRE_THROWS_AS(t.at({2, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(t.at({0, 3}), std::out_of_range);
}

TEST_CASE("Tensor at() wrong number of indices throws", "[tensor][access]") {
    Tensor t(Tensor::Shape{2, 3});
    REQUIRE_THROWS_AS(t.at({0}),       std::invalid_argument);
    REQUIRE_THROWS_AS(t.at({0, 0, 0}), std::invalid_argument);
}

// ─── Static Factories ────────────────────────────────────────────────────────

TEST_CASE("Tensor::zeros", "[tensor][factories]") {
    auto t = Tensor::zeros(Tensor::Shape{3, 4});
    REQUIRE(t.size() == 12);
    REQUIRE(t.sum() == 0.0);
}

TEST_CASE("Tensor::ones", "[tensor][factories]") {
    auto t = Tensor::ones(Tensor::Shape{2, 5});
    REQUIRE(t.sum() == 10.0);
}

TEST_CASE("Tensor::random_uniform range", "[tensor][factories]") {
    seed_random(42);
    auto t = Tensor::random_uniform(Tensor::Shape{10, 10}, -1.0, 1.0);
    REQUIRE(t.min() >= -1.0);
    REQUIRE(t.max() <=  1.0);
}

TEST_CASE("Tensor::random_uniform low > high throws", "[tensor][factories]") {
    REQUIRE_THROWS_AS(Tensor::random_uniform(Tensor::Shape{2, 2}, 1.0, 0.0), std::invalid_argument);
}

TEST_CASE("Tensor::random_normal shape", "[tensor][factories]") {
    seed_random(42);
    auto t = Tensor::random_normal(Tensor::Shape{3, 4, 5});
    REQUIRE(t.rank() == 3);
    REQUIRE(t.size() == 60);
}

TEST_CASE("Tensor::random_normal negative stddev throws", "[tensor][factories]") {
    REQUIRE_THROWS_AS(Tensor::random_normal(Tensor::Shape{2, 2}, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("Tensor seeded random is reproducible", "[tensor][factories]") {
    seed_random(77);
    auto t1 = Tensor::random_uniform(Tensor::Shape{4, 4});
    seed_random(77);
    auto t2 = Tensor::random_uniform(Tensor::Shape{4, 4});
    REQUIRE(t1 == t2);
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

TEST_CASE("Tensor operator+", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b(Tensor::Shape{2, 2}, {5.0, 6.0, 7.0, 8.0});
    auto c = a + b;
    REQUIRE(c[0] == 6.0);
    REQUIRE(c[3] == 12.0);
    REQUIRE(a[0] == 1.0); // original unchanged
}

TEST_CASE("Tensor operator-", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{1, 4}, {5.0, 6.0, 7.0, 8.0});
    Tensor b(Tensor::Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto c = a - b;
    REQUIRE(c[0] == 4.0);
    REQUIRE(c[3] == 4.0);
}

TEST_CASE("Tensor scalar multiply", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{1, 3}, {1.0, 2.0, 3.0});
    auto b = a * 3.0;
    REQUIRE(b[0] == 3.0);
    REQUIRE(b[2] == 9.0);
    auto c = 3.0 * a;
    REQUIRE(c[0] == 3.0);
}

TEST_CASE("Tensor scalar divide", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{1, 4}, {4.0, 6.0, 8.0, 10.0});
    auto b = a / 2.0;
    REQUIRE(b[0] == 2.0);
    REQUIRE(b[3] == 5.0);
}

TEST_CASE("Tensor divide by zero throws", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{2}, 1.0);
    REQUIRE_THROWS_AS(a / 0.0,  std::invalid_argument);
    REQUIRE_THROWS_AS(a /= 0.0, std::invalid_argument);
}

TEST_CASE("Tensor unary negation", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{1, 3}, {1.0, -2.0, 3.0});
    auto b = -a;
    REQUIRE(b[0] == -1.0);
    REQUIRE(b[1] ==  2.0);
    REQUIRE(b[2] == -3.0);
    REQUIRE(a[0] ==  1.0); // original unchanged
}

TEST_CASE("Tensor hadamard product", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    Tensor b(Tensor::Shape{1, 4}, {5.0, 6.0, 7.0, 8.0});
    auto c = a.hadamard(b);
    REQUIRE(c[0] ==  5.0);
    REQUIRE(c[1] == 12.0);
    REQUIRE(c[2] == 21.0);
    REQUIRE(c[3] == 32.0);
}

TEST_CASE("Tensor arithmetic shape mismatch throws", "[tensor][arithmetic]") {
    Tensor a(Tensor::Shape{2, 2});
    Tensor b(Tensor::Shape{2, 3});
    REQUIRE_THROWS_AS(a + b,         std::invalid_argument);
    REQUIRE_THROWS_AS(a - b,         std::invalid_argument);
    REQUIRE_THROWS_AS(a.hadamard(b), std::invalid_argument);
}

// ─── Reshape and flatten ─────────────────────────────────────────────────────

TEST_CASE("Tensor reshape preserves data", "[tensor][shape]") {
    Tensor a(Tensor::Shape{2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = a.reshape(Tensor::Shape{3, 4});
    REQUIRE(b.rank()     == 2);
    REQUIRE(b.dim(0)     == 3);
    REQUIRE(b.dim(1)     == 4);
    REQUIRE(b.size()     == 12);
    REQUIRE(b[0]  == 1.0);
    REQUIRE(b[11] == 12.0);
}

TEST_CASE("Tensor reshape 2D to 3D", "[tensor][shape]") {
    Tensor a(Tensor::Shape{6, 4});
    auto b = a.reshape(Tensor::Shape{2, 3, 4});
    REQUIRE(b.rank()  == 3);
    REQUIRE(b.size()  == 24);
}

TEST_CASE("Tensor reshape size mismatch throws", "[tensor][shape]") {
    Tensor a(Tensor::Shape{2, 3});
    REQUIRE_THROWS_AS(a.reshape(Tensor::Shape{2, 4}), std::invalid_argument);
}

TEST_CASE("Tensor flatten produces 1D tensor", "[tensor][shape]") {
    Tensor a(Tensor::Shape{2, 3, 4});
    auto b = a.flatten();
    REQUIRE(b.rank()   == 1);
    REQUIRE(b.dim(0)   == 24);
    REQUIRE(b.size()   == 24);
}

TEST_CASE("Tensor flatten preserves values", "[tensor][shape]") {
    Tensor a(Tensor::Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = a.flatten();
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE(b[i] == a[i]);
}

// ─── Reductions ──────────────────────────────────────────────────────────────

TEST_CASE("Tensor sum", "[tensor][reductions]") {
    Tensor t(Tensor::Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(t.sum() == 21.0);
}

TEST_CASE("Tensor mean", "[tensor][reductions]") {
    Tensor t(Tensor::Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(t.mean() == 3.5);
}

TEST_CASE("Tensor min", "[tensor][reductions]") {
    Tensor t(Tensor::Shape{1, 5}, {3.0, 1.0, 4.0, 1.0, 5.0});
    REQUIRE(t.min() == 1.0);
}

TEST_CASE("Tensor max", "[tensor][reductions]") {
    Tensor t(Tensor::Shape{1, 5}, {3.0, 1.0, 4.0, 1.0, 9.0});
    REQUIRE(t.max() == 9.0);
}

TEST_CASE("Tensor frobenius_norm", "[tensor][reductions]") {
    Tensor t(Tensor::Shape{1, 3}, {3.0, 4.0, 0.0});
    REQUIRE_THAT(t.frobenius_norm(), WithinAbs(5.0, 1e-9));
}

TEST_CASE("Tensor mean on empty throws", "[tensor][reductions]") {
    REQUIRE_THROWS_AS(Tensor(Tensor::Shape{0}).mean(), std::logic_error);
}

TEST_CASE("Tensor min on empty throws", "[tensor][reductions]") {
    REQUIRE_THROWS_AS(Tensor(Tensor::Shape{0}).min(), std::logic_error);
}

TEST_CASE("Tensor max on empty throws", "[tensor][reductions]") {
    REQUIRE_THROWS_AS(Tensor(Tensor::Shape{0}).max(), std::logic_error);
}

// ─── Apply ───────────────────────────────────────────────────────────────────

TEST_CASE("Tensor apply ReLU", "[tensor][apply]") {
    Tensor t(Tensor::Shape{1, 4}, {-2.0, -1.0, 0.0, 3.0});
    auto r = t.apply([](double x) { return x > 0.0 ? x : 0.0; });
    REQUIRE(r[0] == 0.0);
    REQUIRE(r[1] == 0.0);
    REQUIRE(r[2] == 0.0);
    REQUIRE(r[3] == 3.0);
}

TEST_CASE("Tensor apply does not modify original", "[tensor][apply]") {
    Tensor t(Tensor::Shape{1, 3}, {1.0, 2.0, 3.0});
    (void)t.apply([](double x) { return x * 99.0; });
    REQUIRE(t[0] == 1.0);
}

// ─── Matrix interop ──────────────────────────────────────────────────────────

TEST_CASE("Tensor from_matrix", "[tensor][interop]") {
    Matrix m(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto t = Tensor::from_matrix(m);
    REQUIRE(t.rank()     == 2);
    REQUIRE(t.dim(0)     == 2);
    REQUIRE(t.dim(1)     == 3);
    REQUIRE(t(0, 0)      == 1.0);
    REQUIRE(t(1, 2)      == 6.0);
}

TEST_CASE("Tensor to_matrix on rank-2", "[tensor][interop]") {
    Tensor t(Tensor::Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix m = t.to_matrix();
    REQUIRE(m.rows()   == 2);
    REQUIRE(m.cols()   == 3);
    REQUIRE(m(0, 0)    == 1.0);
    REQUIRE(m(1, 2)    == 6.0);
}

TEST_CASE("Tensor to_matrix round-trip", "[tensor][interop]") {
    Matrix m(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto t = Tensor::from_matrix(m);
    auto m2 = t.to_matrix();
    REQUIRE(m == m2);
}

TEST_CASE("Tensor to_matrix on non-rank-2 throws", "[tensor][interop]") {
    Tensor t(Tensor::Shape{2, 3, 4});
    REQUIRE_THROWS_AS(t.to_matrix(), std::invalid_argument);
}

// ─── Comparison ──────────────────────────────────────────────────────────────

TEST_CASE("Tensor operator== equal", "[tensor][comparison]") {
    Tensor a(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(a == b);
}

TEST_CASE("Tensor operator!= different values", "[tensor][comparison]") {
    Tensor a(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b(Tensor::Shape{2, 2}, {1.0, 2.0, 3.0, 5.0});
    REQUIRE(a != b);
}

TEST_CASE("Tensor operator!= different shapes", "[tensor][comparison]") {
    Tensor a(Tensor::Shape{2, 3});
    Tensor b(Tensor::Shape{3, 2});
    REQUIRE(a != b);
}

TEST_CASE("Tensor approx_equal within tolerance", "[tensor][comparison]") {
    Tensor a(Tensor::Shape{1, 3}, {1.0, 2.0, 3.0});
    Tensor b(Tensor::Shape{1, 3}, {1.0 + 1e-10, 2.0, 3.0});
    REQUIRE(a.approx_equal(b));
}

TEST_CASE("Tensor approx_equal outside tolerance", "[tensor][comparison]") {
    Tensor a(Tensor::Shape{1, 3}, {1.0, 2.0, 3.0});
    Tensor b(Tensor::Shape{1, 3}, {1.0 + 1e-8, 2.0, 3.0});
    REQUIRE_FALSE(a.approx_equal(b));
}

// ─── Cross-type: Matrix seed also seeds Tensor RNG ───────────────────────────

TEST_CASE("seed_random affects both Matrix and Tensor factories", "[tensor][random]") {
    seed_random(55);
    auto m1 = Matrix::random_uniform(2, 2);
    auto t1 = Tensor::random_uniform(Tensor::Shape{2, 2});

    seed_random(55);
    auto m2 = Matrix::random_uniform(2, 2);
    auto t2 = Tensor::random_uniform(Tensor::Shape{2, 2});

    REQUIRE(m1 == m2);
    REQUIRE(t1 == t2);
}
