#pragma once

#include <cstddef>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Dense : public Layer {
public:
    Dense(std::size_t in_features, std::size_t out_features);

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "Dense"; }
    [[nodiscard]] std::size_t param_count() const noexcept override;

    [[nodiscard]] std::size_t in_features()  const noexcept { return in_; }
    [[nodiscard]] std::size_t out_features() const noexcept { return out_; }

    [[nodiscard]] const Matrix& weights() const noexcept { return weights_; }
    [[nodiscard]] const Matrix& bias()    const noexcept { return bias_; }

    void set_weights(Matrix w);
    void set_bias(Matrix b);

private:
    std::size_t in_;
    std::size_t out_;
    Matrix weights_;   // shape (out, in)
    Matrix bias_;      // shape (out, 1)
};

} // namespace neuronix
