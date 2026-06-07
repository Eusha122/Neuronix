#pragma once

#include <random>

// Private header — not installed, not part of the public API.
// C++ guarantees that inline functions with static-local variables
// share a single instance across all translation units, so both
// matrix.cpp and tensor.cpp get the same RNG state.

namespace neuronix::detail {

inline std::mt19937_64& global_rng() {
    static std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

} // namespace neuronix::detail
