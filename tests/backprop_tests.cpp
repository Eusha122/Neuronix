#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "neuronix/layers/dense.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/tanh_activation.hpp"
#include "neuronix/losses/mse_loss.hpp"

using namespace neuronix;
using Catch::Matchers::WithinAbs;

// ── Dense backward ───────────────────────────────────────────────────────────

TEST_CASE("Dense backward - grad_input shape matches input", "[backprop][dense]") {
    Dense d{3, 4};
    Matrix x = Matrix::random_normal(3, 5);
    (void)d.forward_train(x);
    Matrix grad_out = Matrix::ones(4, 5);
    Matrix grad_in = d.backward(grad_out);
    REQUIRE(grad_in.rows() == 3);
    REQUIRE(grad_in.cols() == 5);
}

TEST_CASE("Dense backward - grad_weights shape correct", "[backprop][dense]") {
    Dense d{3, 4};
    Matrix x = Matrix::random_normal(3, 2);
    (void)d.forward_train(x);
    (void)d.backward(Matrix::ones(4, 2));
    REQUIRE(d.grad_weights().rows() == 4);
    REQUIRE(d.grad_weights().cols() == 3);
}

TEST_CASE("Dense backward - grad_bias shape correct", "[backprop][dense]") {
    Dense d{3, 4};
    Matrix x = Matrix::random_normal(3, 2);
    (void)d.forward_train(x);
    (void)d.backward(Matrix::ones(4, 2));
    REQUIRE(d.grad_bias().rows() == 4);
    REQUIRE(d.grad_bias().cols() == 1);
}

TEST_CASE("Dense zero_grad clears gradients", "[backprop][dense]") {
    Dense d{2, 3};
    Matrix x = Matrix::ones(2, 1);
    (void)d.forward_train(x);
    (void)d.backward(Matrix::ones(3, 1));
    d.zero_grad();
    REQUIRE_THAT(d.grad_weights().frobenius_norm(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(d.grad_bias().frobenius_norm(),    WithinAbs(0.0, 1e-9));
}

TEST_CASE("Dense backward - numerical gradient check on weights", "[backprop][dense]") {
    // Verify analytical gradient matches finite-difference estimate.
    Dense d{2, 2};
    d.set_weights(Matrix{2, 2, {0.5, -0.3, 0.8, 0.2}});
    d.set_bias(Matrix::zeros(2, 1));

    Matrix x{2, 1, {1.5, -0.5}};
    Matrix target{2, 1, {1.0, 0.0}};
    MSELoss loss_fn;

    // Analytical gradient
    d.zero_grad();
    Matrix out = d.forward_train(x);
    loss_fn.forward(out, target);
    d.backward(loss_fn.backward());
    Matrix analytical = d.grad_weights();

    // Numerical gradient
    const double eps = 1e-5;
    Matrix numerical = Matrix::zeros(2, 2);
    Matrix w_orig = d.weights();
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            Matrix w_plus = w_orig;
            w_plus(r, c) += eps;
            d.set_weights(w_plus);
            double loss_plus = loss_fn.forward(d.forward(x), target);

            Matrix w_minus = w_orig;
            w_minus(r, c) -= eps;
            d.set_weights(w_minus);
            double loss_minus = loss_fn.forward(d.forward(x), target);

            numerical(r, c) = (loss_plus - loss_minus) / (2.0 * eps);
            d.set_weights(w_orig);
        }
    }

    REQUIRE(analytical.approx_equal(numerical, 1e-6));
}

TEST_CASE("Dense backward - numerical gradient check on bias", "[backprop][dense]") {
    Dense d{2, 2};
    d.set_weights(Matrix::identity(2));
    d.set_bias(Matrix{2, 1, {0.3, -0.2}});

    Matrix x{2, 1, {1.0, 2.0}};
    Matrix target{2, 1, {0.0, 1.0}};
    MSELoss loss_fn;

    d.zero_grad();
    loss_fn.forward(d.forward_train(x), target);
    d.backward(loss_fn.backward());
    Matrix analytical = d.grad_bias();

    const double eps = 1e-5;
    Matrix numerical = Matrix::zeros(2, 1);
    Matrix b_orig = d.bias();
    for (std::size_t r = 0; r < 2; ++r) {
        Matrix b_plus = b_orig; b_plus(r, 0) += eps;
        d.set_bias(b_plus);
        double lp = loss_fn.forward(d.forward(x), target);

        Matrix b_minus = b_orig; b_minus(r, 0) -= eps;
        d.set_bias(b_minus);
        double lm = loss_fn.forward(d.forward(x), target);

        numerical(r, 0) = (lp - lm) / (2.0 * eps);
        d.set_bias(b_orig);
    }

    REQUIRE(analytical.approx_equal(numerical, 1e-6));
}

TEST_CASE("Dense backward - grad accumulates across steps", "[backprop][dense]") {
    Dense d{2, 2};
    d.set_weights(Matrix::identity(2));
    d.set_bias(Matrix::zeros(2, 1));

    Matrix x{2, 1, {1.0, 1.0}};
    Matrix target{2, 1, {0.0, 0.0}};
    MSELoss loss_fn;

    d.zero_grad();
    loss_fn.forward(d.forward_train(x), target);
    d.backward(loss_fn.backward());
    Matrix grad1 = d.grad_weights();

    // Second backward accumulates on top
    loss_fn.forward(d.forward_train(x), target);
    d.backward(loss_fn.backward());
    Matrix grad2 = d.grad_weights();

    REQUIRE(grad2.approx_equal(2.0 * grad1));
}

TEST_CASE("Dense update moves weights toward minimum", "[backprop][dense]") {
    // Single Dense: y = W*x + b, minimise MSE. Weights should decrease toward target.
    Dense d{1, 1};
    d.set_weights(Matrix{1, 1, {2.0}});
    d.set_bias(Matrix::zeros(1, 1));

    Matrix x{1, 1, {1.0}};
    Matrix target{1, 1, {0.0}};
    MSELoss loss_fn;

    double loss_before = loss_fn.forward(d.forward(x), target);

    d.zero_grad();
    loss_fn.forward(d.forward_train(x), target);
    d.backward(loss_fn.backward());
    d.update(0.1);

    double loss_after = loss_fn.forward(d.forward(x), target);
    REQUIRE(loss_after < loss_before);
}

// ── ReLU backward ────────────────────────────────────────────────────────────

TEST_CASE("ReLU backward - zeros gradient where input was negative", "[backprop][relu]") {
    ReLU r;
    Matrix x{4, 1, {-2.0, 1.0, -0.5, 3.0}};
    (void)r.forward_train(x);
    Matrix grad_out = Matrix::ones(4, 1);
    Matrix grad_in = r.backward(grad_out);

    REQUIRE_THAT(grad_in(0, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(grad_in(1, 0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(grad_in(2, 0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(grad_in(3, 0), WithinAbs(1.0, 1e-9));
}

TEST_CASE("ReLU backward - scales gradient by grad_output", "[backprop][relu]") {
    ReLU r;
    Matrix x{2, 1, {1.0, 2.0}};
    (void)r.forward_train(x);
    Matrix grad_out{2, 1, {3.0, 5.0}};
    Matrix grad_in = r.backward(grad_out);

    REQUIRE_THAT(grad_in(0, 0), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(grad_in(1, 0), WithinAbs(5.0, 1e-9));
}

TEST_CASE("ReLU backward preserves shape", "[backprop][relu]") {
    ReLU r;
    Matrix x = Matrix::random_normal(5, 3);
    (void)r.forward_train(x);
    Matrix grad_in = r.backward(Matrix::ones(5, 3));
    REQUIRE(grad_in.rows() == 5);
    REQUIRE(grad_in.cols() == 3);
}

// ── Sigmoid backward ─────────────────────────────────────────────────────────

TEST_CASE("Sigmoid backward - derivative at zero is 0.25", "[backprop][sigmoid]") {
    Sigmoid s;
    Matrix x{1, 1, {0.0}};
    (void)s.forward_train(x);
    Matrix grad_in = s.backward(Matrix{1, 1, {1.0}});
    // sigmoid(0)=0.5, sigmoid'(0)=0.5*(1-0.5)=0.25
    REQUIRE_THAT(grad_in(0, 0), WithinAbs(0.25, 1e-9));
}

TEST_CASE("Sigmoid backward - numerical gradient check", "[backprop][sigmoid]") {
    const double eps = 1e-5;
    Sigmoid s;
    double x_val = 1.2;

    Matrix x_p{1, 1, {x_val + eps}};
    Matrix x_m{1, 1, {x_val - eps}};
    double fp = s.forward(x_p)(0, 0);
    double fm = s.forward(x_m)(0, 0);
    double numerical = (fp - fm) / (2.0 * eps);

    Matrix x_c{1, 1, {x_val}};
    (void)s.forward_train(x_c);
    double analytical = s.backward(Matrix{1, 1, {1.0}})(0, 0);

    REQUIRE_THAT(analytical, WithinAbs(numerical, 1e-6));
}

TEST_CASE("Sigmoid backward preserves shape", "[backprop][sigmoid]") {
    Sigmoid s;
    Matrix x = Matrix::random_normal(4, 2);
    (void)s.forward_train(x);
    Matrix grad_in = s.backward(Matrix::ones(4, 2));
    REQUIRE(grad_in.rows() == 4);
    REQUIRE(grad_in.cols() == 2);
}

// ── Tanh backward ────────────────────────────────────────────────────────────

TEST_CASE("Tanh backward - derivative at zero is 1", "[backprop][tanh_act]") {
    Tanh t;
    Matrix x{1, 1, {0.0}};
    (void)t.forward_train(x);
    Matrix grad_in = t.backward(Matrix{1, 1, {1.0}});
    // tanh'(0) = 1 - tanh(0)^2 = 1
    REQUIRE_THAT(grad_in(0, 0), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Tanh backward - numerical gradient check", "[backprop][tanh_act]") {
    const double eps = 1e-5;
    Tanh t;
    double x_val = 0.8;

    Matrix x_p{1, 1, {x_val + eps}};
    Matrix x_m{1, 1, {x_val - eps}};
    double fp = t.forward(x_p)(0, 0);
    double fm = t.forward(x_m)(0, 0);
    double numerical = (fp - fm) / (2.0 * eps);

    Matrix x_c{1, 1, {x_val}};
    (void)t.forward_train(x_c);
    double analytical = t.backward(Matrix{1, 1, {1.0}})(0, 0);

    REQUIRE_THAT(analytical, WithinAbs(numerical, 1e-6));
}

TEST_CASE("Tanh backward preserves shape", "[backprop][tanh_act]") {
    Tanh t;
    Matrix x = Matrix::random_normal(3, 4);
    (void)t.forward_train(x);
    Matrix grad_in = t.backward(Matrix::ones(3, 4));
    REQUIRE(grad_in.rows() == 3);
    REQUIRE(grad_in.cols() == 4);
}
