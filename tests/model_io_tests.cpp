#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "neuronix/models/model.hpp"
#include "neuronix/losses/mse_loss.hpp"
#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/tanh_activation.hpp"
#include "neuronix/activations/softmax.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// Returns a temp path that is removed when the guard goes out of scope.
struct TempFile {
    std::string path;
    explicit TempFile(const char* name) : path{name} {}
    ~TempFile() { std::filesystem::remove(path); }
};

static Model make_small_model() {
    Model m;
    m.add<Dense>(4, 8);
    m.add<ReLU>();
    m.add<Dense>(8, 3);
    return m;
}

// ── Round-trip: predictions identical ────────────────────────────────────────

TEST_CASE("Model save/load - round-trip predictions match", "[model_io]") {
    TempFile tmp{"__test_model_rt.nnx"};
    seed_random(7);

    Model original = make_small_model();

    Matrix input = Matrix::random_normal(4, 5);
    Matrix pred_before = original.predict(input);

    original.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    Matrix pred_after = loaded.predict(input);

    REQUIRE(pred_before.rows() == pred_after.rows());
    REQUIRE(pred_before.cols() == pred_after.cols());
    for (std::size_t r = 0; r < pred_before.rows(); ++r)
        for (std::size_t c = 0; c < pred_before.cols(); ++c)
            REQUIRE_THAT(pred_after(r, c), WithinAbs(pred_before(r, c), 1e-12));
}

// ── Architecture is preserved ─────────────────────────────────────────────────

TEST_CASE("Model save/load - depth preserved", "[model_io]") {
    TempFile tmp{"__test_model_depth.nnx"};
    Model original = make_small_model();
    original.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    REQUIRE(loaded.depth() == original.depth());
}

TEST_CASE("Model save/load - total_params preserved", "[model_io]") {
    TempFile tmp{"__test_model_params.nnx"};
    Model original = make_small_model();
    original.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    REQUIRE(loaded.total_params() == original.total_params());
}

// ── Weights are bit-exact ─────────────────────────────────────────────────────

TEST_CASE("Model save/load - Dense weights bit-exact", "[model_io]") {
    TempFile tmp{"__test_model_weights.nnx"};
    seed_random(42);
    Model m;
    m.add<Dense>(3, 5);

    // Access the Dense layer through predict (use dynamic_cast on the layer list
    // via a raw-pointer saved before moving into the model).
    // Easier: save, load, and compare predictions column-by-column.
    Matrix X = Matrix::random_normal(3, 10);
    Matrix before = m.predict(X);

    m.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    Matrix after = loaded.predict(X);

    for (std::size_t r = 0; r < before.rows(); ++r)
        for (std::size_t c = 0; c < before.cols(); ++c)
            REQUIRE_THAT(after(r, c), WithinAbs(before(r, c), 0.0));  // exact match
}

// ── All layer types survive round-trip ────────────────────────────────────────

TEST_CASE("Model save/load - all activation types", "[model_io]") {
    TempFile tmp{"__test_model_acts.nnx"};
    seed_random(1);

    Model original;
    original.add<Dense>(4, 4);
    original.add<ReLU>();
    original.add<Dense>(4, 4);
    original.add<Sigmoid>();
    original.add<Dense>(4, 4);
    original.add<Tanh>();
    original.add<Dense>(4, 3);
    original.add<Softmax>();

    Matrix X = Matrix::random_normal(4, 6);
    Matrix pred_before = original.predict(X);

    original.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    Matrix pred_after = loaded.predict(X);

    for (std::size_t r = 0; r < pred_before.rows(); ++r)
        for (std::size_t c = 0; c < pred_before.cols(); ++c)
            REQUIRE_THAT(pred_after(r, c), WithinAbs(pred_before(r, c), 1e-12));
}

// ── Weights survive training then save/load ───────────────────────────────────

TEST_CASE("Model save/load - weights after training", "[model_io]") {
    TempFile tmp{"__test_model_trained.nnx"};
    seed_random(3);

    Model m;
    m.add<Dense>(2, 4);
    m.add<ReLU>();
    m.add<Dense>(4, 1);

    // Train a few steps on simple data
    Matrix X{2, 4, {0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 1.0}};
    Matrix y{1, 4, {0.0, 1.0, 1.0, 0.0}};  // XOR

    neuronix::MSELoss loss_fn;
    for (int i = 0; i < 10; ++i) m.train_step(X, y, loss_fn, 0.01);

    Matrix pred_before = m.predict(X);
    m.save(tmp.path);
    Model loaded = Model::load(tmp.path);
    Matrix pred_after = loaded.predict(X);

    for (std::size_t r = 0; r < pred_before.rows(); ++r)
        for (std::size_t c = 0; c < pred_before.cols(); ++c)
            REQUIRE_THAT(pred_after(r, c), WithinAbs(pred_before(r, c), 1e-12));
}

// ── Error cases ───────────────────────────────────────────────────────────────

TEST_CASE("Model load - missing file throws", "[model_io]") {
    REQUIRE_THROWS_AS(Model::load("__nonexistent_file.nnx"), std::runtime_error);
}

TEST_CASE("Model load - bad magic throws", "[model_io]") {
    TempFile tmp{"__test_model_badmagic.nnx"};
    {
        std::ofstream f{tmp.path, std::ios::binary};
        const char bad[4] = {'B','A','D','!'};
        f.write(bad, 4);
        uint32_t v = 1; f.write(reinterpret_cast<const char*>(&v), 4);
        uint32_t n = 0; f.write(reinterpret_cast<const char*>(&n), 4);
    }
    REQUIRE_THROWS_AS(Model::load(tmp.path), std::runtime_error);
}

TEST_CASE("Model save - bad path throws", "[model_io]") {
    Model m = make_small_model();
    REQUIRE_THROWS_AS(m.save("Z:/no/such/directory/model.nnx"), std::runtime_error);
}
