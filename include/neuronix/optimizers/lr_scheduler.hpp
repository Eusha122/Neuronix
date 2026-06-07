#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace neuronix {

// ── StepLR ───────────────────────────────────────────────────────────────────
// Multiplies the optimizer's learning rate by `gamma` every `step_size` epochs.
// Call scheduler.step() once per epoch, after the optimizer step.
//
//   lr(e) = base_lr * gamma^(floor(e / step_size))
//
template<typename Optimizer>
class StepLR {
public:
    StepLR(Optimizer& opt, std::size_t step_size, double gamma = 0.1)
        : opt_{opt}, step_size_{step_size}, gamma_{gamma}, base_lr_{opt.lr()}
    {
        if (step_size == 0)
            throw std::invalid_argument{"StepLR: step_size must be > 0"};
        if (gamma <= 0.0 || gamma >= 1.0)
            throw std::invalid_argument{"StepLR: gamma must be in (0, 1)"};
    }

    void step() {
        ++epoch_;
        if (epoch_ % step_size_ == 0) {
            double new_lr = opt_.lr() * gamma_;
            opt_.set_lr(new_lr);
        }
    }

    [[nodiscard]] double      lr()    const noexcept { return opt_.lr(); }
    [[nodiscard]] std::size_t epoch() const noexcept { return epoch_; }

    void reset() {
        epoch_ = 0;
        opt_.set_lr(base_lr_);
    }

private:
    Optimizer&  opt_;
    std::size_t step_size_;
    double      gamma_;
    double      base_lr_;
    std::size_t epoch_{0};
};

// ── CosineAnnealingLR ─────────────────────────────────────────────────────────
// Smoothly anneals the learning rate from lr_max (the optimizer's initial lr)
// down to lr_min over T_max epochs, following a half-cosine curve.
// Call scheduler.step() once per epoch, after the optimizer step.
//
//   lr(e) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * e / T_max))
//
template<typename Optimizer>
class CosineAnnealingLR {
public:
    CosineAnnealingLR(Optimizer& opt, std::size_t T_max, double lr_min = 0.0)
        : opt_{opt}, T_max_{T_max}, lr_min_{lr_min}, lr_max_{opt.lr()}
    {
        if (T_max == 0)
            throw std::invalid_argument{"CosineAnnealingLR: T_max must be > 0"};
        if (lr_min < 0.0)
            throw std::invalid_argument{"CosineAnnealingLR: lr_min must be >= 0"};
        if (lr_min >= lr_max_)
            throw std::invalid_argument{"CosineAnnealingLR: lr_min must be < initial lr"};
    }

    void step() {
        ++epoch_;
        const double t   = static_cast<double>(epoch_);
        const double T   = static_cast<double>(T_max_);
        const double new_lr = lr_min_ +
            0.5 * (lr_max_ - lr_min_) * (1.0 + std::cos(std::numbers::pi * t / T));
        opt_.set_lr(new_lr);
    }

    [[nodiscard]] double      lr()    const noexcept { return opt_.lr(); }
    [[nodiscard]] std::size_t epoch() const noexcept { return epoch_; }

    void reset() {
        epoch_ = 0;
        opt_.set_lr(lr_max_);
    }

private:
    Optimizer&  opt_;
    std::size_t T_max_;
    double      lr_min_;
    double      lr_max_;
    std::size_t epoch_{0};
};

} // namespace neuronix
