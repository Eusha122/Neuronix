#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/layers/dense.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

TEST_CASE("Dense constructor sets correct shapes", "[dense]") {
    Dense d{3, 5};
    REQUIRE(d.in_features()  == 3);
    REQUIRE(d.out_features() == 5);
    REQUIRE(d.weights().rows() == 5);
    REQUIRE(d.weights().cols() == 3);
    REQUIRE(d.bias().rows() == 5);
    REQUIRE(d.bias().cols() == 1);
}

TEST_CASE("Dense bias initialised to zero", "[dense]") {
    Dense d{4, 6};
    for (std::size_t r = 0; r < d.bias().rows(); ++r)
        REQUIRE(d.bias()(r, 0) == 0.0);
}

TEST_CASE("Dense weights are non-zero after construction", "[dense]") {
    Dense d{8, 4};
    double norm = d.weights().frobenius_norm();
    REQUIRE(norm > 0.0);
}

TEST_CASE("Dense param_count is correct", "[dense]") {
    Dense d{3, 5};
    REQUIRE(d.param_count() == 3 * 5 + 5);
}

TEST_CASE("Dense name returns Dense", "[dense]") {
    Dense d{2, 2};
    REQUIRE(d.name() == "Dense");
}

TEST_CASE("Dense forward output shape - single sample", "[dense]") {
    Dense d{4, 3};
    Matrix input = Matrix::zeros(4, 1);
    Matrix out = d.forward(input);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 1);
}

TEST_CASE("Dense forward output shape - batched", "[dense]") {
    Dense d{4, 3};
    Matrix input = Matrix::zeros(4, 8);
    Matrix out = d.forward(input);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 8);
}

TEST_CASE("Dense forward with identity weights and zero bias", "[dense]") {
    Dense d{2, 2};
    d.set_weights(Matrix::identity(2));
    d.set_bias(Matrix::zeros(2, 1));

    Matrix input{2, 1, {3.0, 7.0}};
    Matrix out = d.forward(input);

    REQUIRE_THAT(out(0, 0), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(7.0, 1e-9));
}

TEST_CASE("Dense forward with known weights and zero bias", "[dense]") {
    Dense d{2, 3};
    // weights (3x2): [[1,2],[3,4],[5,6]]
    d.set_weights(Matrix{3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});
    d.set_bias(Matrix::zeros(3, 1));

    Matrix input{2, 1, {1.0, 2.0}};
    Matrix out = d.forward(input);

    REQUIRE_THAT(out(0, 0), WithinAbs(5.0,  1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(11.0, 1e-9));
    REQUIRE_THAT(out(2, 0), WithinAbs(17.0, 1e-9));
}

TEST_CASE("Dense forward with known weights and non-zero bias", "[dense]") {
    Dense d{2, 3};
    d.set_weights(Matrix{3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});
    d.set_bias(Matrix{3, 1, {0.5, 1.5, 2.5}});

    Matrix input{2, 1, {1.0, 2.0}};
    Matrix out = d.forward(input);

    REQUIRE_THAT(out(0, 0), WithinAbs(5.5,  1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(12.5, 1e-9));
    REQUIRE_THAT(out(2, 0), WithinAbs(19.5, 1e-9));
}

TEST_CASE("Dense forward batched - bias broadcast to all columns", "[dense]") {
    Dense d{2, 2};
    d.set_weights(Matrix::identity(2));
    d.set_bias(Matrix{2, 1, {1.0, 2.0}});

    // Row-major: row0=[1,2,3], row1=[4,5,6] -> cols: [1,4],[2,5],[3,6]
    Matrix input{2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Matrix out = d.forward(input);

    REQUIRE(out.rows() == 2);
    REQUIRE(out.cols() == 3);
    REQUIRE_THAT(out(0, 0), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(6.0, 1e-9));
    REQUIRE_THAT(out(0, 1), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(out(1, 1), WithinAbs(7.0, 1e-9));
    REQUIRE_THAT(out(0, 2), WithinAbs(4.0, 1e-9));
    REQUIRE_THAT(out(1, 2), WithinAbs(8.0, 1e-9));
}

TEST_CASE("Dense forward throws on input row mismatch", "[dense]") {
    Dense d{4, 2};
    Matrix bad_input = Matrix::zeros(3, 1);
    REQUIRE_THROWS_AS(d.forward(bad_input), std::invalid_argument);
}

TEST_CASE("Dense set_weights throws on shape mismatch", "[dense]") {
    Dense d{3, 5};
    REQUIRE_THROWS_AS(d.set_weights(Matrix::zeros(3, 3)), std::invalid_argument);
}

TEST_CASE("Dense set_bias throws on shape mismatch", "[dense]") {
    Dense d{3, 5};
    REQUIRE_THROWS_AS(d.set_bias(Matrix::zeros(3, 1)), std::invalid_argument);
}

TEST_CASE("Dense set_weights accepts correct shape", "[dense]") {
    Dense d{3, 5};
    Matrix w = Matrix::ones(5, 3);
    REQUIRE_NOTHROW(d.set_weights(w));
    REQUIRE(d.weights().frobenius_norm() > 0.0);
}

TEST_CASE("Dense set_bias accepts correct shape", "[dense]") {
    Dense d{3, 5};
    Matrix b{5, 1, {1.0, 2.0, 3.0, 4.0, 5.0}};
    REQUIRE_NOTHROW(d.set_bias(b));
    REQUIRE_THAT(d.bias()(2, 0), WithinAbs(3.0, 1e-9));
}

TEST_CASE("Dense forward zero input gives bias", "[dense]") {
    Dense d{3, 2};
    d.set_weights(Matrix::zeros(2, 3));
    d.set_bias(Matrix{2, 1, {4.0, 5.0}});

    Matrix input = Matrix::zeros(3, 1);
    Matrix out = d.forward(input);

    REQUIRE_THAT(out(0, 0), WithinAbs(4.0, 1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(5.0, 1e-9));
}
