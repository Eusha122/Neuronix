#include "neuronix/models/model.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace neuronix {

void Model::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Matrix Model::predict(const Matrix& input) const {
    Matrix output = input;
    for (const auto& layer : layers_)
        output = layer->forward(output);
    return output;
}

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
