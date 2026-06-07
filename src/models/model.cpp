#include "neuronix/models/model.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "neuronix/layers/dense.hpp"
#include "neuronix/layers/conv2d.hpp"
#include "neuronix/layers/max_pool2d.hpp"
#include "neuronix/activations/relu.hpp"
#include "neuronix/activations/sigmoid.hpp"
#include "neuronix/activations/tanh_activation.hpp"
#include "neuronix/activations/softmax.hpp"
#include "neuronix/activations/dropout.hpp"
#include "neuronix/layers/batch_norm.hpp"

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

void Model::adam_step(double lr, double beta1, double beta2,
                      double eps, std::size_t t) {
    for (auto& layer : layers_)
        layer->adam_step(lr, beta1, beta2, eps, t);
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

// ── Serialization ────────────────────────────────────────────────────────────

namespace {

enum class LayerTypeId : uint8_t {
    Dense     = 0,
    ReLU      = 1,
    Sigmoid   = 2,
    Tanh      = 3,
    Softmax   = 4,
    Conv2D    = 5,
    MaxPool2D = 6,
    Dropout   = 7,
    BatchNorm = 8,
};

constexpr char     kMagic[4] = {'N','X','N','N'};
constexpr uint32_t kVersion  = 1;

void write_u8(std::ostream& os, uint8_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void write_u32(std::ostream& os, uint32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void write_f64(std::ostream& os, double v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void write_matrix(std::ostream& os, const neuronix::Matrix& m) {
    write_u32(os, static_cast<uint32_t>(m.rows()));
    write_u32(os, static_cast<uint32_t>(m.cols()));
    for (std::size_t r = 0; r < m.rows(); ++r)
        for (std::size_t c = 0; c < m.cols(); ++c)
            write_f64(os, m(r, c));
}

uint8_t  read_u8 (std::istream& is) { uint8_t  v; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; }
uint32_t read_u32(std::istream& is) { uint32_t v; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; }
double   read_f64(std::istream& is) { double   v; is.read(reinterpret_cast<char*>(&v), sizeof v); return v; }

neuronix::Matrix read_matrix(std::istream& is) {
    auto rows = static_cast<std::size_t>(read_u32(is));
    auto cols = static_cast<std::size_t>(read_u32(is));
    neuronix::Matrix m = neuronix::Matrix::zeros(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            m(r, c) = read_f64(is);
    return m;
}

} // anonymous namespace

void neuronix::Model::save(const std::string& path) const {
    std::ofstream os{path, std::ios::binary};
    if (!os) throw std::runtime_error{"Model::save: cannot open '" + path + "'"};

    os.write(kMagic, 4);
    write_u32(os, kVersion);
    write_u32(os, static_cast<uint32_t>(layers_.size()));

    for (const auto& layer : layers_) {
        if (auto* d = dynamic_cast<const Dense*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::Dense));
            write_u32(os, static_cast<uint32_t>(d->in_features()));
            write_u32(os, static_cast<uint32_t>(d->out_features()));
            write_matrix(os, d->weights());
            write_matrix(os, d->bias());
        } else if (dynamic_cast<const ReLU*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::ReLU));
        } else if (dynamic_cast<const Sigmoid*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::Sigmoid));
        } else if (dynamic_cast<const Tanh*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::Tanh));
        } else if (dynamic_cast<const Softmax*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::Softmax));
        } else if (auto* cv = dynamic_cast<const Conv2D*>(layer.get())) {
            write_u8(os,  static_cast<uint8_t>(LayerTypeId::Conv2D));
            write_u32(os, static_cast<uint32_t>(cv->in_channels()));
            write_u32(os, static_cast<uint32_t>(cv->out_channels()));
            write_u32(os, static_cast<uint32_t>(cv->kernel_size()));
            write_u32(os, static_cast<uint32_t>(cv->in_height()));
            write_u32(os, static_cast<uint32_t>(cv->in_width()));
            write_u32(os, static_cast<uint32_t>(cv->padding()));
            write_matrix(os, cv->weights());
            write_matrix(os, cv->bias());
        } else if (auto* mp = dynamic_cast<const MaxPool2D*>(layer.get())) {
            write_u8(os,  static_cast<uint8_t>(LayerTypeId::MaxPool2D));
            write_u32(os, static_cast<uint32_t>(mp->pool_size()));
            write_u32(os, static_cast<uint32_t>(mp->channels()));
            write_u32(os, static_cast<uint32_t>(mp->in_height()));
            write_u32(os, static_cast<uint32_t>(mp->in_width()));
        } else if (auto* dr = dynamic_cast<const Dropout*>(layer.get())) {
            write_u8(os, static_cast<uint8_t>(LayerTypeId::Dropout));
            write_f64(os, dr->drop_rate());
        } else if (auto* bn = dynamic_cast<const BatchNorm*>(layer.get())) {
            write_u8(os,  static_cast<uint8_t>(LayerTypeId::BatchNorm));
            write_u32(os, static_cast<uint32_t>(bn->num_features()));
            write_f64(os, bn->eps());
            write_f64(os, bn->momentum());
            write_matrix(os, bn->gamma());
            write_matrix(os, bn->beta());
            write_matrix(os, bn->running_mean());
            write_matrix(os, bn->running_var());
        } else {
            throw std::runtime_error{
                "Model::save: unsupported layer '" + std::string{layer->name()} + "'"};
        }
    }

    if (!os) throw std::runtime_error{"Model::save: write error"};
}

neuronix::Model neuronix::Model::load(const std::string& path) {
    std::ifstream is{path, std::ios::binary};
    if (!is) throw std::runtime_error{"Model::load: cannot open '" + path + "'"};

    char magic[4];
    is.read(magic, 4);
    if (magic[0] != 'N' || magic[1] != 'X' || magic[2] != 'N' || magic[3] != 'N')
        throw std::runtime_error{"Model::load: not a Neuronix model file"};

    uint32_t version = read_u32(is);
    if (version != kVersion)
        throw std::runtime_error{
            "Model::load: unsupported version " + std::to_string(version)};

    uint32_t num_layers = read_u32(is);

    Model model;
    for (uint32_t i = 0; i < num_layers; ++i) {
        auto type_id = static_cast<LayerTypeId>(read_u8(is));
        switch (type_id) {
            case LayerTypeId::Dense: {
                auto in  = static_cast<std::size_t>(read_u32(is));
                auto out = static_cast<std::size_t>(read_u32(is));
                Matrix w = read_matrix(is);
                Matrix b = read_matrix(is);
                auto d = std::make_unique<Dense>(in, out);
                d->set_weights(std::move(w));
                d->set_bias(std::move(b));
                model.add(std::move(d));
                break;
            }
            case LayerTypeId::ReLU:    model.add<ReLU>();    break;
            case LayerTypeId::Sigmoid: model.add<Sigmoid>(); break;
            case LayerTypeId::Tanh:    model.add<Tanh>();    break;
            case LayerTypeId::Softmax: model.add<Softmax>(); break;
            case LayerTypeId::Conv2D: {
                auto in_c = static_cast<std::size_t>(read_u32(is));
                auto out_c= static_cast<std::size_t>(read_u32(is));
                auto ks   = static_cast<std::size_t>(read_u32(is));
                auto ih   = static_cast<std::size_t>(read_u32(is));
                auto iw   = static_cast<std::size_t>(read_u32(is));
                auto pad  = static_cast<std::size_t>(read_u32(is));
                Matrix w  = read_matrix(is);
                Matrix b  = read_matrix(is);
                auto cv   = std::make_unique<Conv2D>(in_c, out_c, ks, ih, iw, pad);
                cv->set_weights(std::move(w));
                cv->set_bias(std::move(b));
                model.add(std::move(cv));
                break;
            }
            case LayerTypeId::MaxPool2D: {
                auto ps = static_cast<std::size_t>(read_u32(is));
                auto ch = static_cast<std::size_t>(read_u32(is));
                auto ih = static_cast<std::size_t>(read_u32(is));
                auto iw = static_cast<std::size_t>(read_u32(is));
                model.add<MaxPool2D>(ps, ch, ih, iw);
                break;
            }
            case LayerTypeId::Dropout: {
                double p = read_f64(is);
                model.add<Dropout>(p);
                break;
            }
            case LayerTypeId::BatchNorm: {
                auto   nf  = static_cast<std::size_t>(read_u32(is));
                double e   = read_f64(is);
                double mom = read_f64(is);
                Matrix g   = read_matrix(is);
                Matrix b   = read_matrix(is);
                Matrix rm  = read_matrix(is);
                Matrix rv  = read_matrix(is);
                auto bn = std::make_unique<BatchNorm>(nf, e, mom);
                bn->set_gamma(std::move(g));
                bn->set_beta(std::move(b));
                bn->set_running_mean(std::move(rm));
                bn->set_running_var(std::move(rv));
                model.add(std::move(bn));
                break;
            }
            default:
                throw std::runtime_error{"Model::load: unknown layer type id"};
        }
    }

    if (!is) throw std::runtime_error{"Model::load: read error or truncated file"};
    return model;
}

} // namespace neuronix
