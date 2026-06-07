#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/models/model.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/losses/cross_entropy_loss.hpp"
#include "neuronix/optimizers/sgd.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Model training API ────────────────────────────────────────────────────────

TEST_CASE("Model train_forward output shape is correct", "[training]") {
    Model m;
    m.add<Dense>(2, 3);
    m.add<ReLU>();
    Matrix x = Matrix::random_normal(2, 4);
    Matrix out = m.train_forward(x);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 4);
}

TEST_CASE("Model train_step returns finite non-negative loss", "[training]") {
    Model m;
    m.add<Dense>(2, 1);
    MSELoss loss_fn;
    Matrix X = Matrix::random_normal(2, 5);
    Matrix y = Matrix::zeros(1, 5);
    double loss = m.train_step(X, y, loss_fn, 0.01);
    REQUIRE(std::isfinite(loss));
    REQUIRE(loss >= 0.0);
}

TEST_CASE("Model train_step with lr=0 does not change weights", "[training]") {
    auto d = std::make_unique<Dense>(2, 1);
    d->set_weights(Matrix{1, 2, {1.0, 2.0}});
    d->set_bias(Matrix::zeros(1, 1));

    Model m;
    m.add(std::move(d));

    MSELoss loss_fn;
    Matrix X = Matrix::random_normal(2, 4);
    Matrix y = Matrix::zeros(1, 4);

    Matrix pred_before = m.predict(X);
    m.train_step(X, y, loss_fn, 0.0);  // lr=0 → no weight change
    Matrix pred_after = m.predict(X);

    REQUIRE(pred_before.approx_equal(pred_after));
}

// ── SGD on simple problems ────────────────────────────────────────────────────

TEST_CASE("SGD reduces loss on single-weight linear regression", "[training]") {
    auto d = std::make_unique<Dense>(1, 1);
    d->set_weights(Matrix{1, 1, {0.0}});
    d->set_bias(Matrix::zeros(1, 1));

    Model m;
    m.add(std::move(d));

    MSELoss loss_fn;
    SGD opt{m, 0.1};

    // y = 3x
    Matrix X{1, 4, {1.0, 2.0, 3.0, 4.0}};
    Matrix y{1, 4, {3.0, 6.0, 9.0, 12.0}};

    double loss0 = loss_fn.forward(m.predict(X), y);
    for (int i = 0; i < 100; ++i) {
        opt.zero_grad();
        loss_fn.forward(m.train_forward(X), y);
        m.backward(loss_fn.backward());
        opt.step();
    }
    double loss1 = loss_fn.forward(m.predict(X), y);
    REQUIRE(loss1 < loss0);
}

TEST_CASE("SGD set_lr changes learning rate", "[training]") {
    Model m;
    m.add<Dense>(1, 1);
    SGD opt{m, 0.01};
    REQUIRE_THAT(opt.lr(), WithinAbs(0.01, 1e-9));
    opt.set_lr(0.1);
    REQUIRE_THAT(opt.lr(), WithinAbs(0.1, 1e-9));
}

// ── Model::fit convergence ────────────────────────────────────────────────────

TEST_CASE("Model fit converges on constant target", "[training]") {
    auto d = std::make_unique<Dense>(1, 1);
    d->set_weights(Matrix::zeros(1, 1));
    d->set_bias(Matrix::zeros(1, 1));

    Model m;
    m.add(std::move(d));

    MSELoss loss_fn;
    Matrix X = Matrix::ones(1, 8);
    Matrix y{1, 8, {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0}};

    m.fit(X, y, loss_fn, 500, 0.1);

    double loss = loss_fn.forward(m.predict(X), y);
    REQUIRE(loss < 0.01);
}

TEST_CASE("Model fit loss decreases over training", "[training]") {
    seed_random(42);

    Model m;
    m.add<Dense>(2, 4);
    m.add<ReLU>();
    m.add<Dense>(4, 1);

    MSELoss loss_fn;

    seed_random(7);
    Matrix X = Matrix::random_uniform(2, 10, -1.0, 1.0);
    Matrix y = Matrix::random_uniform(1, 10, -1.0, 1.0);

    double loss_start = loss_fn.forward(m.predict(X), y);
    m.fit(X, y, loss_fn, 300, 0.01);
    double loss_end = loss_fn.forward(m.predict(X), y);

    REQUIRE(loss_end < loss_start);
}

TEST_CASE("Model fit learns y=2x", "[training]") {
    auto d = std::make_unique<Dense>(1, 1);
    d->set_weights(Matrix{1, 1, {0.1}});
    d->set_bias(Matrix::zeros(1, 1));

    Model m;
    m.add(std::move(d));

    MSELoss loss_fn;
    Matrix X{1, 6, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Matrix y{1, 6, {2.0, 4.0, 6.0, 8.0, 10.0, 12.0}};

    m.fit(X, y, loss_fn, 2000, 0.01);

    Matrix pred = m.predict(X);
    REQUIRE_THAT(pred(0, 0), WithinAbs(2.0,  0.2));
    REQUIRE_THAT(pred(0, 5), WithinAbs(12.0, 0.5));
}

// ── CrossEntropy training ─────────────────────────────────────────────────────

TEST_CASE("CrossEntropy training loss decreases on 2-class problem", "[training]") {
    seed_random(123);

    Model m;
    m.add<Dense>(2, 4);
    m.add<ReLU>();
    m.add<Dense>(4, 2);

    CrossEntropyLoss loss_fn;

    Matrix X{2, 4, {1.0, -1.0,  1.0, -1.0,
                    0.5,  0.5, -0.5, -0.5}};
    // Class 0 when col0 > 0, class 1 otherwise
    Matrix y{2, 4, {1.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 1.0}};

    double loss0 = loss_fn.forward(m.predict(X), y);
    m.fit(X, y, loss_fn, 500, 0.05);
    double loss1 = loss_fn.forward(m.predict(X), y);

    REQUIRE(loss1 < loss0);
}

// ── zero_grad ─────────────────────────────────────────────────────────────────

TEST_CASE("Model zero_grad resets all gradients", "[training]") {
    Dense* d_ptr;
    {
        auto d = std::make_unique<Dense>(2, 2);
        d->set_weights(Matrix::identity(2));
        d->set_bias(Matrix::zeros(2, 1));
        d_ptr = d.get();

        Model m;
        m.add(std::move(d));

        MSELoss loss_fn;
        Matrix X{2, 1, {1.0, 1.0}};
        Matrix y = Matrix::zeros(2, 1);

        // Accumulate gradients
        loss_fn.forward(m.train_forward(X), y);
        m.backward(loss_fn.backward());

        REQUIRE(d_ptr->grad_weights().frobenius_norm() > 0.0);

        m.zero_grad();

        REQUIRE_THAT(d_ptr->grad_weights().frobenius_norm(), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(d_ptr->grad_bias().frobenius_norm(),    WithinAbs(0.0, 1e-9));
    }
}
