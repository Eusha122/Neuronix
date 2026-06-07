#pragma once

#include "neuronix/core/matrix.hpp"

namespace neuronix {

class Loss {
public:
    virtual ~Loss() = default;
    virtual double forward(const Matrix& predicted, const Matrix& target) = 0;
    [[nodiscard]] virtual Matrix backward() const = 0;
};

} // namespace neuronix
