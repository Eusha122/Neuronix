#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "neuronix/layers/layer.hpp"

namespace neuronix {

class Model {
public:
    void add(std::unique_ptr<Layer> layer);

    template<typename L, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_unique<L>(std::forward<Args>(args)...));
    }

    [[nodiscard]] Matrix predict(const Matrix& input) const;

    [[nodiscard]] std::size_t depth() const noexcept { return layers_.size(); }
    [[nodiscard]] std::size_t total_params() const noexcept;

    void summary() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace neuronix
