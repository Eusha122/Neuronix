#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/models/model.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/softmax.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

TEST_CASE("Empty model predict returns input unchanged", "[model]") {
    Model m;
    Matrix input{3, 1, {1.0, 2.0, 3.0}};
    Matrix out = m.predict(input);
    REQUIRE(out.approx_equal(input));
}

TEST_CASE("Model depth reflects layer count", "[model]") {
    Model m;
    REQUIRE(m.depth() == 0);
    m.add<Dense>(2, 3);
    REQUIRE(m.depth() == 1);
    m.add<ReLU>();
    REQUIRE(m.depth() == 2);
}

TEST_CASE("Model total_params sums layer params", "[model]") {
    Model m;
    m.add<Dense>(4, 8);   // 4*8 + 8 = 40
    m.add<ReLU>();        // 0
    m.add<Dense>(8, 2);   // 8*2 + 2 = 18
    REQUIRE(m.total_params() == 58);
}

TEST_CASE("Model add via unique_ptr", "[model]") {
    Model m;
    m.add(std::make_unique<Dense>(3, 5));
    m.add(std::make_unique<ReLU>());
    REQUIRE(m.depth() == 2);
}

TEST_CASE("Model single Dense layer forward shape", "[model]") {
    Model m;
    m.add<Dense>(4, 3);
    Matrix input = Matrix::zeros(4, 1);
    Matrix out = m.predict(input);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 1);
}

TEST_CASE("Model Dense then ReLU forward shape", "[model]") {
    Model m;
    m.add<Dense>(4, 6);
    m.add<ReLU>();
    Matrix input = Matrix::zeros(4, 1);
    Matrix out = m.predict(input);
    REQUIRE(out.rows() == 6);
    REQUIRE(out.cols() == 1);
}

TEST_CASE("Model ReLU zeros out all-negative Dense outputs", "[model]") {
    auto d = std::make_unique<Dense>(2, 3);
    d->set_weights(Matrix{3, 2, {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}});
    d->set_bias(Matrix::zeros(3, 1));

    Model m;
    m.add(std::move(d));
    m.add<ReLU>();

    Matrix input{2, 1, {1.0, 1.0}};
    Matrix out = m.predict(input);
    for (std::size_t r = 0; r < out.rows(); ++r)
        REQUIRE_THAT(out(r, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Model two Dense layers - correct numerical result", "[model]") {
    auto d1 = std::make_unique<Dense>(2, 2);
    d1->set_weights(Matrix::identity(2));
    d1->set_bias(Matrix::zeros(2, 1));

    auto d2 = std::make_unique<Dense>(2, 2);
    d2->set_weights(Matrix{2, 2, {2.0, 0.0, 0.0, 2.0}});
    d2->set_bias(Matrix::zeros(2, 1));

    Model m;
    m.add(std::move(d1));
    m.add(std::move(d2));

    Matrix input{2, 1, {3.0, 5.0}};
    Matrix out = m.predict(input);

    REQUIRE_THAT(out(0, 0), WithinAbs(6.0,  1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(10.0, 1e-9));
}

TEST_CASE("Model Dense-ReLU-Dense pipeline - known result", "[model]") {
    // layer 1 (2→3): W=[[1,0],[0,1],[-1,-1]], b=0
    // input [2,3] → after W: [2, 3, -5] → after ReLU: [2, 3, 0]
    // layer 2 (3→1): W=[[1,1,1]], b=0 → output: 5
    auto d1 = std::make_unique<Dense>(2, 3);
    d1->set_weights(Matrix{3, 2, {1.0, 0.0, 0.0, 1.0, -1.0, -1.0}});
    d1->set_bias(Matrix::zeros(3, 1));

    auto d2 = std::make_unique<Dense>(3, 1);
    d2->set_weights(Matrix{1, 3, {1.0, 1.0, 1.0}});
    d2->set_bias(Matrix::zeros(1, 1));

    Model m;
    m.add(std::move(d1));
    m.add<ReLU>();
    m.add(std::move(d2));

    Matrix input{2, 1, {2.0, 3.0}};
    Matrix out = m.predict(input);

    REQUIRE(out.rows() == 1);
    REQUIRE(out.cols() == 1);
    REQUIRE_THAT(out(0, 0), WithinAbs(5.0, 1e-9));
}

TEST_CASE("Model with Softmax output sums to 1", "[model]") {
    Model m;
    m.add<Dense>(3, 5);
    m.add<Softmax>();

    Matrix input = Matrix::random_normal(3, 1);
    Matrix out = m.predict(input);

    REQUIRE(out.rows() == 5);
    REQUIRE_THAT(out.sum(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Model batched predict preserves batch dimension", "[model]") {
    auto d = std::make_unique<Dense>(3, 2);
    d->set_weights(Matrix{2, 3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}});
    d->set_bias(Matrix::zeros(2, 1));

    Model m;
    m.add(std::move(d));

    Matrix input = Matrix::ones(3, 5);
    Matrix out = m.predict(input);

    REQUIRE(out.rows() == 2);
    REQUIRE(out.cols() == 5);
}

TEST_CASE("Model summary does not throw", "[model]") {
    Model m;
    m.add<Dense>(784, 128);
    m.add<ReLU>();
    m.add<Dense>(128, 10);
    m.add<Softmax>();
    REQUIRE_NOTHROW(m.summary());
}

TEST_CASE("Model with Sigmoid output in (0, 1)", "[model]") {
    Model m;
    m.add<Dense>(4, 1);
    m.add<Sigmoid>();

    Matrix input = Matrix::random_normal(4, 1);
    Matrix out = m.predict(input);

    REQUIRE(out(0, 0) > 0.0);
    REQUIRE(out(0, 0) < 1.0);
}

TEST_CASE("Model Dense-Sigmoid-Dense pipeline shape", "[model]") {
    Model m;
    m.add<Dense>(5, 8);
    m.add<Sigmoid>();
    m.add<Dense>(8, 3);
    m.add<Softmax>();

    Matrix input = Matrix::random_normal(5, 1);
    Matrix out = m.predict(input);

    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 1);
    REQUIRE_THAT(out.sum(), WithinAbs(1.0, 1e-9));
}
