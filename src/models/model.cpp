#include "neuronix/models/model.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace neuronix {

void Model::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

// ── Inference ────────────────────────────────────────────────────────────────

Matrix Model::predict(const Matrix& input) const {
    Matrix output = input;
    for (const auto& layer : layers_)
        output = layer->forward(output);
    return output;
}

// ── Training ─────────────────────────────────────────────────────────────────

Matrix Model::train_forward(const Matrix& input) {
    Matrix output = input;
    for (auto& layer : layers_)
        output = layer->forward_train(output);
    return output;
}

void Model::backward(const Matrix& grad_output) {
    Matrix grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = (*it)->backward(grad);
}

void Model::update(double lr) {
    for (auto& layer : layers_)
        layer->update(lr);
}

void Model::zero_grad() {
    for (auto& layer : layers_)
        layer->zero_grad();
}

double Model::train_step(const Matrix& X, const Matrix& y, Loss& loss_fn, double lr) {
    zero_grad();
    Matrix output = train_forward(X);
    double loss = loss_fn.forward(output, y);
    backward(loss_fn.backward());
    update(lr);
    return loss;
}

void Model::fit(const Matrix& X, const Matrix& y, Loss& loss_fn,
                std::size_t epochs, double lr,
                bool verbose, std::size_t print_every) {
    for (std::size_t e = 0; e < epochs; ++e) {
        double loss = train_step(X, y, loss_fn, lr);
        if (verbose && (e % print_every == 0 || e == epochs - 1)) {
            std::cout << "Epoch " << std::setw(5) << e
                      << "  loss: " << loss << '\n';
        }
    }
}

// ── Metadata ──────────────────────────────────────────────────────────────────

std::size_t Model::total_params() const noexcept {
    std::size_t total = 0;
    for (const auto& layer : layers_)
        total += layer->param_count();
    return total;
}

void Model::summary() const {
    std::cout << "Model Summary\n";
    std::cout << std::string(36, '=') << '\n';
    std::cout << std::left << std::setw(20) << "Layer"
              << std::setw(10) << "Params" << '\n';
    std::cout << std::string(36, '-') << '\n';
    for (const auto& layer : layers_) {
        std::cout << std::setw(20) << layer->name()
                  << std::setw(10) << layer->param_count() << '\n';
    }
    std::cout << std::string(36, '=') << '\n';
    std::cout << "Total params: " << total_params() << '\n';
}

} // namespace neuronix
