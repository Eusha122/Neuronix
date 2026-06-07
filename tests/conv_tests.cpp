#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "neuronix/layers/conv2d.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/layers/max_pool2d.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/models/model.hpp"
#include "neuronix/optimizers/adam.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Conv2D shape tests ────────────────────────────────────────────────────────

TEST_CASE("Conv2D output shape - no padding", "[conv2d]") {
    Conv2D c{1, 4, 3, 8, 8};         // (1,8,8) → (4,6,6)
    Matrix X = Matrix::zeros(1*8*8, 5);
    Matrix out = c.forward(X);
    REQUIRE(out.rows() == 4*6*6);
    REQUIRE(out.cols() == 5);
}

TEST_CASE("Conv2D output shape - same padding", "[conv2d]") {
    Conv2D c{1, 4, 3, 8, 8, 1};     // (1,8,8) → (4,8,8) with pad=1
    Matrix X = Matrix::zeros(1*8*8, 3);
    Matrix out = c.forward(X);
    REQUIRE(out.rows() == 4*8*8);
    REQUIRE(out.cols() == 3);
}

TEST_CASE("Conv2D output shape - 1x1 kernel", "[conv2d]") {
    Conv2D c{3, 8, 1, 16, 16};      // (3,16,16) → (8,16,16)
    Matrix X = Matrix::zeros(3*16*16, 2);
    Matrix out = c.forward(X);
    REQUIRE(out.rows() == 8*16*16);
    REQUIRE(out.cols() == 2);
}

TEST_CASE("Conv2D accessors", "[conv2d]") {
    Conv2D c{2, 5, 3, 10, 10, 1};
    REQUIRE(c.in_channels()  == 2);
    REQUIRE(c.out_channels() == 5);
    REQUIRE(c.kernel_size()  == 3);
    REQUIRE(c.in_height()    == 10);
    REQUIRE(c.in_width()     == 10);
    REQUIRE(c.out_height()   == 10);  // same padding
    REQUIRE(c.out_width()    == 10);
    REQUIRE(c.padding()      == 1);
    REQUIRE(c.param_count()  == 5*2*3*3 + 5);
}

TEST_CASE("Conv2D param_count", "[conv2d]") {
    Conv2D c{1, 6, 5, 28, 28};
    REQUIRE(c.param_count() == 6*1*5*5 + 6);   // 156
}

// ── Known-value forward test ──────────────────────────────────────────────────

TEST_CASE("Conv2D 1x1 kernel with weight=1 bias=0 is identity", "[conv2d]") {
    // 1 input channel, 1 output channel, 1x1 kernel, 3x3 input → 3x3 output
    Conv2D c{1, 1, 1, 3, 3};

    // Force weights = [[1]] and bias = [0]
    Matrix w{1, 1, {1.0}};
    Matrix b{1, 1, {0.0}};
    c.set_weights(w);
    c.set_bias(b);

    Matrix X{9, 1, {1,2,3,4,5,6,7,8,9}};
    Matrix out = c.forward(X);

    REQUIRE(out.rows() == 9);
    REQUIRE(out.cols() == 1);
    for (std::size_t r = 0; r < 9; ++r)
        REQUIRE_THAT(out(r, 0), WithinAbs(X(r, 0), 1e-12));
}

TEST_CASE("Conv2D 2x2 kernel on 2x2 input - single filter", "[conv2d]") {
    // 1 channel, 2x2 input, 2x2 kernel → 1x1 output
    Conv2D c{1, 1, 2, 2, 2};

    // Weights: [[1,0],[0,1]]  bias: 0
    // → dot product with input [[a,b],[c,d]] = a*1 + b*0 + c*0 + d*1 = a+d
    Matrix w{1, 4, {1.0, 0.0, 0.0, 1.0}};
    Matrix b{1, 1, {0.0}};
    c.set_weights(w);
    c.set_bias(b);

    // Input: [[1,2],[3,4]] → a+d = 1+4 = 5
    Matrix X{4, 1, {1.0, 2.0, 3.0, 4.0}};
    Matrix out = c.forward(X);

    REQUIRE(out.rows() == 1);
    REQUIRE(out.cols() == 1);
    REQUIRE_THAT(out(0, 0), WithinAbs(5.0, 1e-12));
}

// ── Backward shape ────────────────────────────────────────────────────────────

TEST_CASE("Conv2D backward returns correct input gradient shape", "[conv2d]") {
    seed_random(1);
    Conv2D c{2, 4, 3, 6, 6};
    const std::size_t N = 3;
    Matrix X = Matrix::random_normal(2*6*6, N);
    c.forward_train(X);

    Matrix g = Matrix::ones(4*4*4, N);
    Matrix dx = c.backward(g);
    REQUIRE(dx.rows() == 2*6*6);
    REQUIRE(dx.cols() == N);
}

// ── Numerical gradient check ──────────────────────────────────────────────────

TEST_CASE("Conv2D backward numerical gradient check", "[conv2d]") {
    seed_random(42);
    const double eps = 1e-5;

    // Small conv to keep the check fast
    Conv2D c{1, 2, 2, 4, 4};   // (1,4,4) → (2,3,3), N=2
    const std::size_t N = 2;
    Matrix X = Matrix::random_normal(1*4*4, N);
    Matrix target = Matrix::zeros(2*3*3, N);

    MSELoss loss_fn;

    // Analytical gradients
    c.zero_grad();
    Matrix out = c.forward_train(X);
    [[maybe_unused]] auto l0 = loss_fn.forward(out, target);
    c.backward(loss_fn.backward());
    Matrix ag = c.grad_weights();

    // Numerical gradients for a sample of weight elements
    Matrix w_orig = c.weights();
    const std::size_t check_n = std::min(std::size_t{8}, ag.rows() * ag.cols());
    for (std::size_t idx = 0; idx < check_n; ++idx) {
        std::size_t r = idx / ag.cols();
        std::size_t col = idx % ag.cols();

        Matrix w_plus = w_orig;
        w_plus(r, col) += eps;
        c.set_weights(w_plus);
        double loss_plus = loss_fn.forward(c.forward(X), target);

        Matrix w_minus = w_orig;
        w_minus(r, col) -= eps;
        c.set_weights(w_minus);
        double loss_minus = loss_fn.forward(c.forward(X), target);

        double numerical = (loss_plus - loss_minus) / (2.0 * eps);
        REQUIRE_THAT(ag(r, col), WithinAbs(numerical, 1e-4));
    }
    c.set_weights(w_orig);
}

// ── Conv2D rejects invalid args ───────────────────────────────────────────────

TEST_CASE("Conv2D rejects kernel larger than padded input", "[conv2d]") {
    REQUIRE_THROWS_AS((Conv2D{1, 1, 5, 3, 3}), std::invalid_argument);
}

// ── MaxPool2D shape tests ─────────────────────────────────────────────────────

TEST_CASE("MaxPool2D output shape", "[maxpool2d]") {
    MaxPool2D mp{2, 4, 8, 8};
    Matrix X = Matrix::zeros(4*8*8, 5);
    Matrix out = mp.forward(X);
    REQUIRE(out.rows() == 4*4*4);
    REQUIRE(out.cols() == 5);
}

TEST_CASE("MaxPool2D accessors", "[maxpool2d]") {
    MaxPool2D mp{2, 6, 28, 28};
    REQUIRE(mp.pool_size()  == 2);
    REQUIRE(mp.channels()   == 6);
    REQUIRE(mp.in_height()  == 28);
    REQUIRE(mp.in_width()   == 28);
    REQUIRE(mp.out_height() == 14);
    REQUIRE(mp.out_width()  == 14);
}

TEST_CASE("MaxPool2D forward picks max in each window", "[maxpool2d]") {
    // 1 channel, 4x4 input, pool=2 → 2x2 output
    // Input:  [[1,3,2,4],
    //          [5,2,6,1],   top-left pool → max(1,3,5,2) = 5
    //          [0,1,3,2],   top-right pool → max(2,4,6,1) = 6
    //          [2,3,1,4]]   bottom-left → max(0,1,2,3) = 3
    //                       bottom-right → max(3,2,1,4) = 4
    MaxPool2D mp{2, 1, 4, 4};
    Matrix X{16, 1, {1,3,2,4,
                      5,2,6,1,
                      0,1,3,2,
                      2,3,1,4}};
    Matrix out = mp.forward(X);
    REQUIRE(out.rows() == 4);
    REQUIRE_THAT(out(0, 0), WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(out(1, 0), WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(out(2, 0), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(out(3, 0), WithinAbs(4.0, 1e-12));
}

TEST_CASE("MaxPool2D backward routes gradient to max position", "[maxpool2d]") {
    // Same 4x4 input as above
    MaxPool2D mp{2, 1, 4, 4};
    Matrix X{16, 1, {1,3,2,4,
                      5,2,6,1,
                      0,1,3,2,
                      2,3,1,4}};
    [[maybe_unused]] auto mp_out = mp.forward_train(X);

    Matrix grad_out = Matrix::ones(4, 1);   // all-ones upstream gradient
    Matrix grad_in  = mp.backward(grad_out);

    REQUIRE(grad_in.rows() == 16);
    // Max of top-left pool was at index 4 (value 5)
    REQUIRE_THAT(grad_in(4,  0), WithinAbs(1.0, 1e-12));
    // Max of top-right pool was at index 6 (value 6)
    REQUIRE_THAT(grad_in(6,  0), WithinAbs(1.0, 1e-12));
    // Non-max positions in each pool should be 0
    REQUIRE_THAT(grad_in(0,  0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(grad_in(1,  0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("MaxPool2D rejects non-divisible size", "[maxpool2d]") {
    REQUIRE_THROWS_AS((MaxPool2D{2, 1, 5, 4}), std::invalid_argument);
}

// ── End-to-end: Conv → MaxPool → Dense trains ────────────────────────────────

TEST_CASE("Conv2D + MaxPool2D + Dense trains without error", "[conv2d][maxpool2d]") {
    seed_random(7);

    // Input: (1, 8, 8), 4 samples
    // Conv(1,4,3,8,8) → (4,6,6)
    // MaxPool(2,4,6,6) → (4,3,3) = 36 features
    // Dense(36,10) → (10, N)
    Model m;
    m.add<Conv2D>(1, 4, 3, 8, 8);
    m.add<ReLU>();
    m.add<MaxPool2D>(2, 4, 6, 6);
    m.add<Dense>(36, 10);

    const std::size_t N = 4;
    Matrix X = Matrix::random_normal(64, N);
    Matrix y = Matrix::zeros(10, N);
    for (std::size_t n = 0; n < N; ++n) y(n % 10, n) = 1.0;

    MSELoss loss_fn;
    Adam    opt{m, 0.01};

    double first_loss = loss_fn.forward(m.predict(X), y);
    for (int i = 0; i < 20; ++i) {
        opt.zero_grad();
        auto out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        opt.step();
    }
    double final_loss = loss_fn.forward(m.predict(X), y);
    REQUIRE(final_loss < first_loss);
}
