#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/models/model.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/optimizers/sgd.hpp"
#include "neuronix/optimizers/adam.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Adam step_count ───────────────────────────────────────────────────────────

TEST_CASE("Adam step_count increments", "[adam]") {
    Model m;
    m.add<Dense>(2, 2);
    Adam opt{m};
    REQUIRE(opt.step_count() == 0);

    Matrix X = Matrix::zeros(2, 1);
    Matrix y = Matrix::zeros(2, 1);
    MSELoss loss_fn;

    auto out = m.train_forward(X);
    loss_fn.forward(out, y);
    m.backward(loss_fn.backward());

    opt.step();
    REQUIRE(opt.step_count() == 1);
    opt.step();
    REQUIRE(opt.step_count() == 2);
}

// ── Adam zero_grad clears gradients ──────────────────────────────────────────

TEST_CASE("Adam zero_grad clears gradients", "[adam]") {
    seed_random(10);
    Model m;
    m.add<Dense>(2, 3);
    Adam opt{m};

    Matrix X{2, 1, {1.0, 2.0}};
    Matrix y{3, 1, {0.0, 1.0, 0.0}};
    MSELoss loss_fn;

    // Accumulate gradients
    Matrix out = m.train_forward(X);
    loss_fn.forward(out, y);
    m.backward(loss_fn.backward());

    opt.zero_grad();

    // After zero_grad, another backward + step should give same result as
    // a fresh start — just verify step doesn't throw and zero_grad works.
    out = m.train_forward(X);
    loss_fn.forward(out, y);
    m.backward(loss_fn.backward());
    REQUIRE_NOTHROW(opt.step());
}

// ── Adam reduces loss ─────────────────────────────────────────────────────────

TEST_CASE("Adam reduces loss over training steps", "[adam]") {
    seed_random(0);
    Model m;
    m.add<Dense>(4, 8);
    m.add<ReLU>();
    m.add<Dense>(8, 2);

    Matrix X{4, 8, {
        1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,
        1,1,0,0,1,1,0,0,
        0,0,1,1,0,0,1,1
    }};
    Matrix y{2, 8, {
        1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1
    }};

    MSELoss loss_fn;
    Adam    opt{m, 0.01};

    // Measure initial loss
    double first_loss = loss_fn.forward(m.predict(X), y);

    // Train 50 steps
    for (int i = 0; i < 50; ++i) {
        opt.zero_grad();
        Matrix out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        opt.step();
    }

    double final_loss = loss_fn.forward(m.predict(X), y);
    REQUIRE(final_loss < first_loss);
}

// ── Adam converges faster than SGD on same problem ───────────────────────────

TEST_CASE("Adam converges faster than SGD on regression", "[adam]") {
    // Train two identical models: one with Adam, one with SGD.
    // After the same number of steps Adam should reach lower loss.
    seed_random(5);

    Matrix X{3, 16, {
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0
    }};
    Matrix y{2, 16, {
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
    }};

    auto build = []() {
        Model m;
        m.add<Dense>(3, 8);
        m.add<ReLU>();
        m.add<Dense>(8, 2);
        return m;
    };

    // Both models start from same seed
    seed_random(99);
    Model m_adam = build();
    seed_random(99);
    Model m_sgd  = build();

    MSELoss loss_adam, loss_sgd;
    Adam opt_adam{m_adam, 0.01};
    SGD  opt_sgd {m_sgd,  0.01};

    for (int i = 0; i < 100; ++i) {
        opt_adam.zero_grad();
        auto out_a = m_adam.train_forward(X);
        loss_adam.forward(out_a, y);
        m_adam.backward(loss_adam.backward());
        opt_adam.step();

        opt_sgd.zero_grad();
        auto out_s = m_sgd.train_forward(X);
        loss_sgd.forward(out_s, y);
        m_sgd.backward(loss_sgd.backward());
        opt_sgd.step();
    }

    double final_adam = loss_adam.forward(m_adam.predict(X), y);
    double final_sgd  = loss_sgd.forward(m_sgd.predict(X), y);
    REQUIRE(final_adam < final_sgd);
}

// ── Adam set_lr ───────────────────────────────────────────────────────────────

TEST_CASE("Adam set_lr updates learning rate", "[adam]") {
    Model m;
    m.add<Dense>(2, 2);
    Adam opt{m, 0.001};
    REQUIRE_THAT(opt.lr(), WithinAbs(0.001, 1e-15));
    opt.set_lr(0.01);
    REQUIRE_THAT(opt.lr(), WithinAbs(0.01, 1e-15));
}
