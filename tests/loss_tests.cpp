#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/losses/cross_entropy_loss.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── MSELoss ──────────────────────────────────────────────────────────────────

TEST_CASE("MSELoss forward - identical inputs give zero loss", "[loss][mse]") {
    MSELoss l;
    Matrix x{3, 1, {1.0, 2.0, 3.0}};
    REQUIRE_THAT(l.forward(x, x), WithinAbs(0.0, 1e-9));
}

TEST_CASE("MSELoss forward - known value", "[loss][mse]") {
    MSELoss l;
    // diff = [1, 1], mse = mean([1, 1]) = 1
    Matrix pred{2, 1, {2.0, 3.0}};
    Matrix tgt {2, 1, {1.0, 2.0}};
    REQUIRE_THAT(l.forward(pred, tgt), WithinAbs(1.0, 1e-9));
}

TEST_CASE("MSELoss forward - non-negative", "[loss][mse]") {
    MSELoss l;
    Matrix pred = Matrix::random_normal(4, 3);
    Matrix tgt  = Matrix::random_normal(4, 3);
    REQUIRE(l.forward(pred, tgt) >= 0.0);
}

TEST_CASE("MSELoss backward - shape matches predicted", "[loss][mse]") {
    MSELoss l;
    Matrix pred{3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Matrix tgt  = Matrix::zeros(3, 2);
    l.forward(pred, tgt);
    Matrix grad = l.backward();
    REQUIRE(grad.rows() == 3);
    REQUIRE(grad.cols() == 2);
}

TEST_CASE("MSELoss backward - gradient direction", "[loss][mse]") {
    // pred > target → grad should be positive
    MSELoss l;
    Matrix pred{1, 1, {3.0}};
    Matrix tgt {1, 1, {1.0}};
    l.forward(pred, tgt);
    REQUIRE(l.backward()(0, 0) > 0.0);
}

TEST_CASE("MSELoss backward - known gradient value", "[loss][mse]") {
    // pred=[3], tgt=[1], diff=2, loss=4, grad=2*2/1=4
    MSELoss l;
    Matrix pred{1, 1, {3.0}};
    Matrix tgt {1, 1, {1.0}};
    l.forward(pred, tgt);
    REQUIRE_THAT(l.backward()(0, 0), WithinAbs(4.0, 1e-9));
}

TEST_CASE("MSELoss shape mismatch throws", "[loss][mse]") {
    MSELoss l;
    REQUIRE_THROWS_AS(
        l.forward(Matrix::zeros(2, 1), Matrix::zeros(3, 1)),
        std::invalid_argument);
}

// ── CrossEntropyLoss ──────────────────────────────────────────────────────────

TEST_CASE("CrossEntropyLoss forward - non-negative", "[loss][ce]") {
    CrossEntropyLoss l;
    Matrix logits{3, 2, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0}};
    Matrix target{3, 2, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    REQUIRE(l.forward(logits, target) >= 0.0);
}

TEST_CASE("CrossEntropyLoss forward - perfect prediction is low loss", "[loss][ce]") {
    CrossEntropyLoss l;
    // Large logit for correct class → softmax ≈ [1,0,0] → low CE
    Matrix logits{3, 1, {100.0, 0.0, 0.0}};
    Matrix target{3, 1, {1.0,   0.0, 0.0}};
    REQUIRE(l.forward(logits, target) < 0.01);
}

TEST_CASE("CrossEntropyLoss forward - uniform logits give log(num_classes)", "[loss][ce]") {
    CrossEntropyLoss l;
    // Uniform logits → softmax = [0.25, 0.25, 0.25, 0.25]
    // CE = -log(0.25) = log(4) ≈ 1.386
    Matrix logits = Matrix::zeros(4, 1);
    Matrix target{4, 1, {1.0, 0.0, 0.0, 0.0}};
    REQUIRE_THAT(l.forward(logits, target), WithinAbs(std::log(4.0), 1e-6));
}

TEST_CASE("CrossEntropyLoss backward - shape matches logits", "[loss][ce]") {
    CrossEntropyLoss l;
    Matrix logits = Matrix::zeros(4, 3);
    Matrix target = Matrix::zeros(4, 3);
    target(0, 0) = 1.0; target(1, 1) = 1.0; target(2, 2) = 1.0;
    l.forward(logits, target);
    Matrix grad = l.backward();
    REQUIRE(grad.rows() == 4);
    REQUIRE(grad.cols() == 3);
}

TEST_CASE("CrossEntropyLoss backward - each column sums near zero", "[loss][ce]") {
    // sum(softmax - onehot) = 1 - 1 = 0 per column
    CrossEntropyLoss l;
    Matrix logits{3, 2, {1.0, 0.5, 2.0, 1.5, 0.5, 2.0}};
    Matrix target{3, 2, {1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    l.forward(logits, target);
    Matrix grad = l.backward();
    for (std::size_t c = 0; c < grad.cols(); ++c) {
        double col_sum = 0.0;
        for (std::size_t r = 0; r < grad.rows(); ++r)
            col_sum += grad(r, c);
        REQUIRE_THAT(col_sum, WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("CrossEntropyLoss shape mismatch throws", "[loss][ce]") {
    CrossEntropyLoss l;
    REQUIRE_THROWS_AS(
        l.forward(Matrix::zeros(3, 1), Matrix::zeros(4, 1)),
        std::invalid_argument);
}
