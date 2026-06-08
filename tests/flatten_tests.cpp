#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>

#include "neuronix/layers/flatten.hpp"
#include "neuronix/layers/conv2d.hpp"
#include "neuronix/layers/max_pool2d.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/models/model.hpp"
#include "neuronix/optimizers/adam.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

TEST_CASE("Flatten forward is identity", "[flatten]") {
    Flatten f;
    Matrix X{3, 4, {1,2,3,4, 5,6,7,8, 9,10,11,12}};
    Matrix out = f.forward(X);
    REQUIRE(out.rows() == 3);
    REQUIRE(out.cols() == 4);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            REQUIRE_THAT(out(r, c), WithinAbs(X(r, c), 1e-12));
}

TEST_CASE("Flatten forward_train is identity", "[flatten]") {
    Flatten f;
    Matrix X = Matrix::random_normal(6, 10);
    Matrix out = f.forward_train(X);
    REQUIRE(out.rows() == X.rows());
    REQUIRE(out.cols() == X.cols());
    for (std::size_t r = 0; r < X.rows(); ++r)
        for (std::size_t c = 0; c < X.cols(); ++c)
            REQUIRE_THAT(out(r, c), WithinAbs(X(r, c), 1e-12));
}

TEST_CASE("Flatten backward is identity", "[flatten]") {
    Flatten f;
    Matrix g = Matrix::random_normal(5, 8);
    Matrix dx = f.backward(g);
    REQUIRE(dx.rows() == g.rows());
    REQUIRE(dx.cols() == g.cols());
    for (std::size_t r = 0; r < g.rows(); ++r)
        for (std::size_t c = 0; c < g.cols(); ++c)
            REQUIRE_THAT(dx(r, c), WithinAbs(g(r, c), 1e-12));
}

TEST_CASE("Flatten has zero params", "[flatten]") {
    Flatten f;
    REQUIRE(f.param_count() == 0);
}

TEST_CASE("Flatten name is Flatten", "[flatten]") {
    Flatten f;
    REQUIRE(f.name() == "Flatten");
}

TEST_CASE("Flatten save/load round-trip", "[flatten]") {
    Model m;
    m.add<Conv2D>(1, 4, 3, 8, 8);
    m.add<ReLU>();
    m.add<MaxPool2D>(2, 4, 6, 6);
    m.add<Flatten>();
    m.add<Dense>(36, 5);

    seed_random(1);
    Matrix X = Matrix::random_normal(64, 4);
    Matrix pred_before = m.predict(X);

    m.save("__test_flatten.nnx");
    Model loaded = Model::load("__test_flatten.nnx");
    std::filesystem::remove("__test_flatten.nnx");

    Matrix pred_after = loaded.predict(X);
    for (std::size_t r = 0; r < pred_before.rows(); ++r)
        for (std::size_t c = 0; c < pred_before.cols(); ++c)
            REQUIRE_THAT(pred_after(r, c), WithinAbs(pred_before(r, c), 1e-12));
}

TEST_CASE("Conv2D + Flatten + Dense trains end-to-end", "[flatten]") {
    seed_random(5);
    Model m;
    m.add<Conv2D>(1, 4, 3, 8, 8);   // (1,8,8) → (4,6,6)
    m.add<ReLU>();
    m.add<MaxPool2D>(2, 4, 6, 6);   // → (4,3,3) = 36 features
    m.add<Flatten>();
    m.add<Dense>(36, 4);

    const std::size_t N = 8;
    Matrix X = Matrix::random_normal(64, N);
    Matrix y = Matrix::zeros(4, N);
    for (std::size_t n = 0; n < N; ++n) y(n % 4, n) = 1.0;

    MSELoss loss_fn;
    Adam    opt{m, 0.01};

    double first = loss_fn.forward(m.predict(X), y);
    for (int i = 0; i < 30; ++i) {
        opt.zero_grad();
        auto out = m.train_forward(X);
        loss_fn.forward(out, y);
        m.backward(loss_fn.backward());
        opt.step();
    }
    REQUIRE(loss_fn.forward(m.predict(X), y) < first);
}
