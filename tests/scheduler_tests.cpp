#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/models/model.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/optimizers/sgd.hpp"
#include "neuronix/optimizers/adam.hpp"
#include "neuronix/optimizers/lr_scheduler.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── StepLR ────────────────────────────────────────────────────────────────────

TEST_CASE("StepLR does not change lr before step_size epochs", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    StepLR sched{opt, 3, 0.1};

    sched.step(); REQUIRE_THAT(sched.lr(), WithinAbs(0.1, 1e-12));
    sched.step(); REQUIRE_THAT(sched.lr(), WithinAbs(0.1, 1e-12));
}

TEST_CASE("StepLR reduces lr at step_size boundary", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    StepLR sched{opt, 3, 0.1};

    sched.step(); sched.step(); sched.step();
    REQUIRE_THAT(sched.lr(), WithinAbs(0.01, 1e-12));
}

TEST_CASE("StepLR reduces lr at each multiple of step_size", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 1.0};
    StepLR sched{opt, 2, 0.5};

    sched.step(); sched.step();                          // epoch 2 → 0.5
    REQUIRE_THAT(sched.lr(), WithinAbs(0.5, 1e-12));
    sched.step(); sched.step();                          // epoch 4 → 0.25
    REQUIRE_THAT(sched.lr(), WithinAbs(0.25, 1e-12));
    sched.step(); sched.step();                          // epoch 6 → 0.125
    REQUIRE_THAT(sched.lr(), WithinAbs(0.125, 1e-12));
}

TEST_CASE("StepLR epoch counter increments", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    StepLR sched{opt, 5, 0.1};
    sched.step(); sched.step(); sched.step();
    REQUIRE(sched.epoch() == 3);
}

TEST_CASE("StepLR reset restores base lr", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    StepLR sched{opt, 2, 0.1};
    sched.step(); sched.step();                          // lr is now 0.01
    sched.reset();
    REQUIRE_THAT(sched.lr(),    WithinAbs(0.1, 1e-12));
    REQUIRE(sched.epoch() == 0);
}

TEST_CASE("StepLR works with Adam", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    Adam opt{m, 0.001};
    StepLR<Adam> sched{opt, 2, 0.1};
    sched.step(); sched.step();
    REQUIRE_THAT(sched.lr(), WithinAbs(0.0001, 1e-15));
}

TEST_CASE("StepLR rejects step_size 0", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    REQUIRE_THROWS_AS((StepLR{opt, 0, 0.5}), std::invalid_argument);
}

TEST_CASE("StepLR rejects gamma >= 1", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    REQUIRE_THROWS_AS((StepLR{opt, 2, 1.0}), std::invalid_argument);
    REQUIRE_THROWS_AS((StepLR{opt, 2, 1.5}), std::invalid_argument);
}

TEST_CASE("StepLR rejects gamma <= 0", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    REQUIRE_THROWS_AS((StepLR{opt, 2, 0.0}), std::invalid_argument);
}

// ── CosineAnnealingLR ─────────────────────────────────────────────────────────

TEST_CASE("CosineAnnealingLR starts at lr_max", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    Adam opt{m, 0.01};
    CosineAnnealingLR sched{opt, 10, 0.0001};
    // Before any step, lr is still the optimizer's initial lr
    REQUIRE_THAT(sched.lr(), WithinAbs(0.01, 1e-12));
}

TEST_CASE("CosineAnnealingLR reaches lr_min at T_max", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    Adam opt{m, 0.01};
    CosineAnnealingLR sched{opt, 10, 0.0001};

    for (int i = 0; i < 10; ++i) sched.step();
    REQUIRE_THAT(sched.lr(), WithinAbs(0.0001, 1e-12));
}

TEST_CASE("CosineAnnealingLR is monotonically decreasing", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 1.0};
    CosineAnnealingLR sched{opt, 20, 0.0};

    double prev = sched.lr();
    for (int i = 0; i < 20; ++i) {
        sched.step();
        REQUIRE(sched.lr() <= prev);
        prev = sched.lr();
    }
}

TEST_CASE("CosineAnnealingLR epoch counter increments", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    CosineAnnealingLR sched{opt, 10};
    sched.step(); sched.step();
    REQUIRE(sched.epoch() == 2);
}

TEST_CASE("CosineAnnealingLR reset restores lr_max", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    CosineAnnealingLR sched{opt, 5, 0.001};
    for (int i = 0; i < 5; ++i) sched.step();
    sched.reset();
    REQUIRE_THAT(sched.lr(),    WithinAbs(0.1, 1e-12));
    REQUIRE(sched.epoch() == 0);
}

TEST_CASE("CosineAnnealingLR rejects T_max 0", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    REQUIRE_THROWS_AS((CosineAnnealingLR{opt, 0}), std::invalid_argument);
}

TEST_CASE("CosineAnnealingLR rejects lr_min >= lr_max", "[scheduler]") {
    Model m; m.add<Dense>(2, 2);
    SGD opt{m, 0.1};
    REQUIRE_THROWS_AS((CosineAnnealingLR{opt, 10, 0.1}),  std::invalid_argument);
    REQUIRE_THROWS_AS((CosineAnnealingLR{opt, 10, 0.5}),  std::invalid_argument);
}
