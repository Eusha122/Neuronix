#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/activations/dropout.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/core/matrix.hpp"
#include "neuronix/models/model.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/optimizers/adam.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Constructor validation ────────────────────────────────────────────────────

TEST_CASE("Dropout rejects drop_rate >= 1", "[dropout]") {
    REQUIRE_THROWS_AS(Dropout{1.0},  std::invalid_argument);
    REQUIRE_THROWS_AS(Dropout{1.5},  std::invalid_argument);
}

TEST_CASE("Dropout rejects negative drop_rate", "[dropout]") {
    REQUIRE_THROWS_AS(Dropout{-0.1}, std::invalid_argument);
}

TEST_CASE("Dropout accepts drop_rate 0", "[dropout]") {
    REQUIRE_NOTHROW(Dropout{0.0});
}

// ── Inference passthrough ─────────────────────────────────────────────────────

TEST_CASE("Dropout forward is identity (inference)", "[dropout]") {
    Dropout d{0.5};
    Matrix X{2, 4, {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0}};
    Matrix out = d.forward(X);
    REQUIRE(out.rows() == X.rows());
    REQUIRE(out.cols() == X.cols());
    for (std::size_t r = 0; r < X.rows(); ++r)
        for (std::size_t c = 0; c < X.cols(); ++c)
            REQUIRE_THAT(out(r, c), WithinAbs(X(r, c), 1e-12));
}

// ── Training forward drops units ──────────────────────────────────────────────

TEST_CASE("Dropout forward_train output has same shape", "[dropout]") {
    seed_random(1);
    Dropout d{0.3};
    Matrix X = Matrix::ones(4, 100);
    Matrix out = d.forward_train(X);
    REQUIRE(out.rows() == 4);
    REQUIRE(out.cols() == 100);
}

TEST_CASE("Dropout forward_train drops approximately p fraction", "[dropout]") {
    // With a large enough sample the zero fraction should be close to p.
    seed_random(42);
    const double p = 0.5;
    Dropout d{p};
    Matrix X = Matrix::ones(1, 10000);
    Matrix out = d.forward_train(X);

    std::size_t zeros = 0;
    for (std::size_t c = 0; c < out.cols(); ++c)
        if (out(0, c) == 0.0) ++zeros;

    double zero_rate = static_cast<double>(zeros) / 10000.0;
    REQUIRE_THAT(zero_rate, WithinAbs(p, 0.03));  // within 3%
}

TEST_CASE("Dropout forward_train scales kept units by 1/(1-p)", "[dropout]") {
    seed_random(7);
    const double p = 0.4;
    Dropout d{p};
    Matrix X = Matrix::ones(1, 10000);
    Matrix out = d.forward_train(X);

    const double expected_scale = 1.0 / (1.0 - p);
    for (std::size_t c = 0; c < out.cols(); ++c) {
        double v = out(0, c);
        REQUIRE((v == 0.0 || std::abs(v - expected_scale) < 1e-12));
    }
}

// ── p=0 is a passthrough in both modes ───────────────────────────────────────

TEST_CASE("Dropout with p=0 forward_train is identity", "[dropout]") {
    seed_random(3);
    Dropout d{0.0};
    Matrix X{2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Matrix out = d.forward_train(X);
    for (std::size_t r = 0; r < X.rows(); ++r)
        for (std::size_t c = 0; c < X.cols(); ++c)
            REQUIRE_THAT(out(r, c), WithinAbs(X(r, c), 1e-12));
}

// ── Backward uses same mask ───────────────────────────────────────────────────

TEST_CASE("Dropout backward zeroes the same positions as forward_train", "[dropout]") {
    seed_random(99);
    Dropout d{0.5};
    Matrix X = Matrix::ones(3, 200);
    Matrix out = d.forward_train(X);

    // Pass all-ones gradient
    Matrix ones_grad = Matrix::ones(3, 200);
    Matrix grad_in   = d.backward(ones_grad);

    // Every position that was zeroed in forward must be zero in grad_in,
    // and every kept position must equal the scale factor.
    const double scale = 1.0 / 0.5;
    for (std::size_t r = 0; r < out.rows(); ++r) {
        for (std::size_t c = 0; c < out.cols(); ++c) {
            if (out(r, c) == 0.0)
                REQUIRE_THAT(grad_in(r, c), WithinAbs(0.0, 1e-12));
            else
                REQUIRE_THAT(grad_in(r, c), WithinAbs(scale, 1e-12));
        }
    }
}

// ── Inverted dropout preserves expected value ─────────────────────────────────

TEST_CASE("Dropout forward_train mean ≈ input mean (inverted dropout)", "[dropout]") {
    seed_random(17);
    Dropout d{0.3};
    Matrix X = Matrix::ones(1, 50000);
    Matrix out = d.forward_train(X);

    double sum = 0.0;
    for (std::size_t c = 0; c < out.cols(); ++c) sum += out(0, c);
    double mean = sum / 50000.0;
    REQUIRE_THAT(mean, WithinAbs(1.0, 0.02));  // should be ≈ 1.0
}

// ── Dropout integrates with Model + Adam ─────────────────────────────────────

TEST_CASE("Dropout inside Model trains without error", "[dropout]") {
    seed_random(5);
    Model m;
    m.add<Dense>(4, 8);
    m.add<ReLU>();
    m.add<Dropout>(0.3);
    m.add<Dense>(8, 2);

    Matrix X{4, 16, {
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,
        0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1
    }};
    Matrix y{2, 16, {
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
    }};

    MSELoss loss_fn;
    Adam    opt{m, 0.01};

    double first_loss = loss_fn.forward(m.predict(X), y);

    for (int i = 0; i < 100; ++i) {
        opt.zero_grad();
        auto out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        opt.step();
    }

    double final_loss = loss_fn.forward(m.predict(X), y);
    REQUIRE(final_loss < first_loss);
}
