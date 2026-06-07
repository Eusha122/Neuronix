#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/softmax.hpp"
#include "neuronix/activations/tanh_activation.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ─── ReLU ────────────────────────────────────────────────────────────────────

TEST_CASE("ReLU name", "[relu]") {
    ReLU r;
    REQUIRE(r.name() == "ReLU");
}

TEST_CASE("ReLU param_count is zero", "[relu]") {
    ReLU r;
    REQUIRE(r.param_count() == 0);
}

TEST_CASE("ReLU clamps negatives to zero", "[relu]") {
    ReLU r;
    Matrix input{3, 1, {-5.0, -0.1, -100.0}};
    Matrix out = r.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(out(2, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("ReLU passes positives unchanged", "[relu]") {
    ReLU r;
    Matrix input{3, 1, {1.0, 3.5, 100.0}};
    Matrix out = r.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(1.0,   1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(3.5,   1e-9));
    REQUIRE_THAT(out(2, 0), WithinAbs(100.0, 1e-9));
}

TEST_CASE("ReLU maps zero to zero", "[relu]") {
    ReLU r;
    Matrix input{1, 1, {0.0}};
    Matrix out = r.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("ReLU mixed positive and negative", "[relu]") {
    ReLU r;
    Matrix input{4, 1, {-2.0, 0.0, 3.0, -0.5}};
    Matrix out = r.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(out(1, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(out(2, 0), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(out(3, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("ReLU does not modify original matrix", "[relu]") {
    ReLU r;
    Matrix input{2, 1, {-1.0, 2.0}};
    (void)r.forward(input);
    REQUIRE_THAT(input(0, 0), WithinAbs(-1.0, 1e-9));
}

TEST_CASE("ReLU preserves shape", "[relu]") {
    ReLU r;
    Matrix input = Matrix::random_normal(5, 4);
    Matrix out = r.forward(input);
    REQUIRE(out.rows() == 5);
    REQUIRE(out.cols() == 4);
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

TEST_CASE("Sigmoid name", "[sigmoid]") {
    Sigmoid s;
    REQUIRE(s.name() == "Sigmoid");
}

TEST_CASE("Sigmoid param_count is zero", "[sigmoid]") {
    Sigmoid s;
    REQUIRE(s.param_count() == 0);
}

TEST_CASE("Sigmoid at zero is 0.5", "[sigmoid]") {
    Sigmoid s;
    Matrix input{1, 1, {0.0}};
    Matrix out = s.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.5, 1e-9));
}

TEST_CASE("Sigmoid output in (0, 1) for moderate inputs", "[sigmoid]") {
    // Use values where floating-point doesn't saturate to exactly 0 or 1
    Sigmoid s;
    Matrix input{4, 1, {-10.0, -1.0, 1.0, 10.0}};
    Matrix out = s.forward(input);
    for (std::size_t r = 0; r < out.rows(); ++r) {
        REQUIRE(out(r, 0) > 0.0);
        REQUIRE(out(r, 0) < 1.0);
    }
}

TEST_CASE("Sigmoid large positive is very close to 1", "[sigmoid]") {
    Sigmoid s;
    Matrix input{1, 1, {500.0}};
    Matrix out = s.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Sigmoid large negative is very close to 0", "[sigmoid]") {
    Sigmoid s;
    Matrix input{1, 1, {-500.0}};
    Matrix out = s.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Sigmoid preserves shape", "[sigmoid]") {
    Sigmoid s;
    Matrix input = Matrix::zeros(3, 7);
    Matrix out = s.forward(input);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 7);
}

// ─── Tanh ────────────────────────────────────────────────────────────────────

TEST_CASE("Tanh name", "[tanh_act]") {
    Tanh t;
    REQUIRE(t.name() == "Tanh");
}

TEST_CASE("Tanh param_count is zero", "[tanh_act]") {
    Tanh t;
    REQUIRE(t.param_count() == 0);
}

TEST_CASE("Tanh at zero is 0", "[tanh_act]") {
    Tanh t;
    Matrix input{1, 1, {0.0}};
    Matrix out = t.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Tanh output in (-1, 1) for moderate inputs", "[tanh_act]") {
    // Use values where floating-point doesn't saturate to exactly -1 or 1
    Tanh t;
    Matrix input{4, 1, {-5.0, -1.0, 1.0, 5.0}};
    Matrix out = t.forward(input);
    for (std::size_t r = 0; r < out.rows(); ++r) {
        REQUIRE(out(r, 0) > -1.0);
        REQUIRE(out(r, 0) <  1.0);
    }
}

TEST_CASE("Tanh is antisymmetric", "[tanh_act]") {
    Tanh t;
    Matrix pos{1, 1, {2.5}};
    Matrix neg{1, 1, {-2.5}};
    Matrix out_pos = t.forward(pos);
    Matrix out_neg = t.forward(neg);
    REQUIRE_THAT(out_pos(0, 0), WithinAbs(-out_neg(0, 0), 1e-9));
}

TEST_CASE("Tanh large positive is very close to 1", "[tanh_act]") {
    Tanh t;
    Matrix input{1, 1, {500.0}};
    Matrix out = t.forward(input);
    REQUIRE_THAT(out(0, 0), WithinAbs(1.0, 1e-9));
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

TEST_CASE("Softmax name", "[softmax]") {
    Softmax s;
    REQUIRE(s.name() == "Softmax");
}

TEST_CASE("Softmax param_count is zero", "[softmax]") {
    Softmax s;
    REQUIRE(s.param_count() == 0);
}

TEST_CASE("Softmax outputs sum to 1 - single column", "[softmax]") {
    Softmax s;
    Matrix input{4, 1, {1.0, 2.0, 3.0, 4.0}};
    Matrix out = s.forward(input);
    REQUIRE_THAT(out.sum(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Softmax all outputs positive", "[softmax]") {
    Softmax s;
    Matrix input{3, 1, {-5.0, 0.0, 5.0}};
    Matrix out = s.forward(input);
    for (std::size_t r = 0; r < out.rows(); ++r)
        REQUIRE(out(r, 0) > 0.0);
}

TEST_CASE("Softmax argmax preserved", "[softmax]") {
    Softmax s;
    Matrix input{4, 1, {1.0, 2.0, 10.0, 3.0}};
    Matrix out = s.forward(input);
    double max_val = out(0, 0);
    std::size_t max_idx = 0;
    for (std::size_t r = 1; r < out.rows(); ++r) {
        if (out(r, 0) > max_val) { max_val = out(r, 0); max_idx = r; }
    }
    REQUIRE(max_idx == 2);
}

TEST_CASE("Softmax uniform input yields equal outputs", "[softmax]") {
    Softmax s;
    Matrix input{4, 1, {2.0, 2.0, 2.0, 2.0}};
    Matrix out = s.forward(input);
    for (std::size_t r = 0; r < out.rows(); ++r)
        REQUIRE_THAT(out(r, 0), WithinAbs(0.25, 1e-9));
}

TEST_CASE("Softmax batched - each column sums to 1", "[softmax]") {
    Softmax s;
    Matrix input{3, 4, {1.0, 2.0, 3.0,  0.5,
                        4.0, 1.0, 2.0,  1.5,
                        2.0, 3.0, 1.0,  2.0}};
    Matrix out = s.forward(input);
    for (std::size_t c = 0; c < out.cols(); ++c) {
        double col_sum = 0.0;
        for (std::size_t r = 0; r < out.rows(); ++r)
            col_sum += out(r, c);
        REQUIRE_THAT(col_sum, WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Softmax numerically stable with large inputs", "[softmax]") {
    Softmax s;
    Matrix input{3, 1, {1000.0, 1001.0, 1002.0}};
    Matrix out = s.forward(input);
    REQUIRE_THAT(out.sum(), WithinAbs(1.0, 1e-9));
    for (std::size_t r = 0; r < out.rows(); ++r)
        REQUIRE(std::isfinite(out(r, 0)));
}

TEST_CASE("Softmax preserves shape", "[softmax]") {
    Softmax s;
    Matrix input = Matrix::ones(5, 3);
    Matrix out = s.forward(input);
    REQUIRE(out.rows() == 5);
    REQUIRE(out.cols() == 3);
}
