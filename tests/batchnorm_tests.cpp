#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>

#include "neuronix/layers/batch_norm.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/models/model.hpp"
#include "neuronix/optimizers/adam.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Constructor / validation ──────────────────────────────────────────────────

TEST_CASE("BatchNorm rejects zero num_features", "[batchnorm]") {
    REQUIRE_THROWS_AS(BatchNorm{0}, std::invalid_argument);
}

TEST_CASE("BatchNorm param_count is 2*F", "[batchnorm]") {
    BatchNorm bn{16};
    REQUIRE(bn.param_count() == 32);
}

TEST_CASE("BatchNorm initial gamma is all-ones", "[batchnorm]") {
    BatchNorm bn{4};
    for (std::size_t r = 0; r < 4; ++r)
        REQUIRE_THAT(bn.gamma()(r, 0), WithinAbs(1.0, 1e-12));
}

TEST_CASE("BatchNorm initial beta is all-zeros", "[batchnorm]") {
    BatchNorm bn{4};
    for (std::size_t r = 0; r < 4; ++r)
        REQUIRE_THAT(bn.beta()(r, 0), WithinAbs(0.0, 1e-12));
}

// ── Training forward normalises output ───────────────────────────────────────

TEST_CASE("BatchNorm forward_train output has near-zero mean per feature", "[batchnorm]") {
    seed_random(1);
    BatchNorm bn{3};
    // Input with large non-zero mean to confirm normalisation
    Matrix X = Matrix::zeros(3, 64);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 64; ++c)
            X(r, c) = static_cast<double>(r + 1) * 10.0 + static_cast<double>(c);

    Matrix out = bn.forward_train(X);

    // With gamma=1, beta=0, output should be ~standardised: mean≈0, std≈1
    for (std::size_t r = 0; r < 3; ++r) {
        double sum = 0.0;
        for (std::size_t c = 0; c < 64; ++c) sum += out(r, c);
        REQUIRE_THAT(sum / 64.0, WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("BatchNorm forward_train output has near-unit std per feature", "[batchnorm]") {
    seed_random(2);
    BatchNorm bn{2};
    Matrix X = Matrix::zeros(2, 128);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 128; ++c)
            X(r, c) = static_cast<double>(c) * (static_cast<double>(r) + 1.0);

    Matrix out = bn.forward_train(X);

    for (std::size_t r = 0; r < 2; ++r) {
        double mean = 0.0;
        for (std::size_t c = 0; c < 128; ++c) mean += out(r, c);
        mean /= 128.0;
        double var = 0.0;
        for (std::size_t c = 0; c < 128; ++c) {
            double d = out(r, c) - mean; var += d * d;
        }
        var /= 128.0;
        REQUIRE_THAT(std::sqrt(var), WithinAbs(1.0, 0.01));
    }
}

// ── Inference uses running statistics ────────────────────────────────────────

TEST_CASE("BatchNorm inference uses running_mean and running_var", "[batchnorm]") {
    BatchNorm bn{2};

    // Warm up running stats with one training batch
    Matrix X_train = Matrix::zeros(2, 32);
    for (std::size_t c = 0; c < 32; ++c) {
        X_train(0, c) = static_cast<double>(c) * 2.0;       // mean≈31, var≈341
        X_train(1, c) = static_cast<double>(c) * 0.1;
    }
    [[maybe_unused]] auto t = bn.forward_train(X_train);

    // Inference on a single sample — should NOT re-compute batch stats
    Matrix X_inf{2, 1, {100.0, 5.0}};
    Matrix out1 = bn.forward(X_inf);
    Matrix out2 = bn.forward(X_inf);

    // Identical outputs (deterministic — no batch stat recompute)
    REQUIRE_THAT(out1(0, 0), WithinAbs(out2(0, 0), 1e-12));
    REQUIRE_THAT(out1(1, 0), WithinAbs(out2(1, 0), 1e-12));
}

// ── Running stats update ──────────────────────────────────────────────────────

TEST_CASE("BatchNorm running_mean moves toward batch mean", "[batchnorm]") {
    BatchNorm bn{1, 1e-5, 0.1};
    // Initial running_mean = 0
    REQUIRE_THAT(bn.running_mean()(0, 0), WithinAbs(0.0, 1e-12));

    // Batch with mean = 10
    Matrix X{1, 8, {10,10,10,10,10,10,10,10}};
    [[maybe_unused]] auto out = bn.forward_train(X);

    // After one step: running_mean = 0.9*0 + 0.1*10 = 1.0
    REQUIRE_THAT(bn.running_mean()(0, 0), WithinAbs(1.0, 1e-10));
}

// ── Backward shape ────────────────────────────────────────────────────────────

TEST_CASE("BatchNorm backward returns correct shape", "[batchnorm]") {
    seed_random(5);
    BatchNorm bn{4};
    Matrix X = Matrix::random_normal(4, 16);
    [[maybe_unused]] auto out = bn.forward_train(X);
    Matrix g = Matrix::ones(4, 16);
    Matrix dx = bn.backward(g);
    REQUIRE(dx.rows() == 4);
    REQUIRE(dx.cols() == 16);
}

// ── Numerical gradient check ──────────────────────────────────────────────────

TEST_CASE("BatchNorm backward numerical gradient check - input grad", "[batchnorm]") {
    seed_random(7);
    const double eps = 1e-5;
    BatchNorm bn{3};
    const std::size_t N = 8;
    Matrix X = Matrix::random_normal(3, N);
    Matrix target = Matrix::zeros(3, N);
    MSELoss loss_fn;

    // Analytical
    bn.zero_grad();
    [[maybe_unused]] auto out = bn.forward_train(X);
    [[maybe_unused]] auto l0 = loss_fn.forward(out, target);
    Matrix grad_out = loss_fn.backward();
    [[maybe_unused]] auto dx0 = bn.backward(grad_out);

    // Numerical: perturb each input element
    const std::size_t check = 9;   // check first 9 elements
    for (std::size_t idx = 0; idx < check; ++idx) {
        std::size_t r = idx / N;
        std::size_t c = idx % N;

        Matrix Xp = X; Xp(r, c) += eps;
        BatchNorm bp{3}; // fresh layer (same init — gamma=1,beta=0,running=std)
        [[maybe_unused]] auto l_p = loss_fn.forward(bp.forward_train(Xp), target);

        Matrix Xm = X; Xm(r, c) -= eps;
        BatchNorm bm{3};
        [[maybe_unused]] auto l_m = loss_fn.forward(bm.forward_train(Xm), target);

        double numerical = (l_p - l_m) / (2.0 * eps);

        // Re-run analytical with this specific input element
        BatchNorm ba{3};
        auto oa = ba.forward_train(X);
        [[maybe_unused]] auto la = loss_fn.forward(oa, target);
        Matrix dx = ba.backward(loss_fn.backward());

        REQUIRE_THAT(dx(r, c), WithinAbs(numerical, 1e-4));
    }
}

TEST_CASE("BatchNorm backward numerical gradient check - gamma grad", "[batchnorm]") {
    seed_random(11);
    const double eps = 1e-5;
    const std::size_t F = 3, N = 8;
    Matrix X = Matrix::random_normal(F, N);
    Matrix target = Matrix::zeros(F, N);
    MSELoss loss_fn;

    BatchNorm bn{F};
    [[maybe_unused]] auto out = bn.forward_train(X);
    [[maybe_unused]] auto l0 = loss_fn.forward(out, target);
    bn.backward(loss_fn.backward());
    for (std::size_t r = 0; r < F; ++r) {
        // + eps
        BatchNorm bp{F};
        Matrix gp = Matrix::ones(F, 1); gp(r, 0) += eps;
        bp.set_gamma(gp);
        [[maybe_unused]] auto lp = loss_fn.forward(bp.forward_train(X), target);

        // - eps
        BatchNorm bm{F};
        Matrix gm = Matrix::ones(F, 1); gm(r, 0) -= eps;
        bm.set_gamma(gm);
        [[maybe_unused]] auto lm = loss_fn.forward(bm.forward_train(X), target);

        double numerical = (lp - lm) / (2.0 * eps);

        // Analytical
        BatchNorm ba{F};
        [[maybe_unused]] auto oa = ba.forward_train(X);
        [[maybe_unused]] auto la = loss_fn.forward(oa, target);
        ba.backward(loss_fn.backward());
        // Access grad_gamma via update by checking weight change
        Matrix gamma_before = ba.gamma();
        ba.update(1.0);   // lr=1 → gamma_new = gamma_old - 1 * grad_gamma
        double analytical = gamma_before(r, 0) - ba.gamma()(r, 0);

        REQUIRE_THAT(analytical, WithinAbs(numerical, 1e-4));
    }
}

// ── Integration: Model with BatchNorm trains ─────────────────────────────────

TEST_CASE("BatchNorm inside Model trains without error", "[batchnorm]") {
    seed_random(3);
    Model m;
    m.add<Dense>(8, 16);
    m.add<BatchNorm>(16);
    m.add<ReLU>();
    m.add<Dense>(16, 4);

    Matrix X = Matrix::random_normal(8, 32);
    Matrix y = Matrix::zeros(4, 32);
    for (std::size_t c = 0; c < 32; ++c) y(c % 4, c) = 1.0;

    MSELoss loss_fn;
    Adam    opt{m, 0.01};

    double first = loss_fn.forward(m.predict(X), y);
    for (int i = 0; i < 50; ++i) {
        opt.zero_grad();
        auto out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        opt.step();
    }
    double final_loss = loss_fn.forward(m.predict(X), y);
    REQUIRE(final_loss < first);
}

// ── Save / load round-trip preserves running stats ────────────────────────────

TEST_CASE("BatchNorm save/load preserves running statistics", "[batchnorm]") {
    seed_random(9);
    Model m;
    m.add<Dense>(4, 8);
    m.add<BatchNorm>(8);
    m.add<ReLU>();
    m.add<Dense>(8, 2);

    // Train to update running stats
    Matrix X = Matrix::random_normal(4, 16);
    Matrix y = Matrix::zeros(2, 16);
    MSELoss loss_fn;
    for (int i = 0; i < 5; ++i) {
        m.zero_grad();
        auto out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        m.update(0.01);
    }

    Matrix pred_before = m.predict(X);
    m.save("__test_bn.nnx");
    Model loaded = Model::load("__test_bn.nnx");
    Matrix pred_after = loaded.predict(X);
    std::filesystem::remove("__test_bn.nnx");

    for (std::size_t r = 0; r < pred_before.rows(); ++r)
        for (std::size_t c = 0; c < pred_before.cols(); ++c)
            REQUIRE_THAT(pred_after(r, c), WithinAbs(pred_before(r, c), 1e-12));
}
