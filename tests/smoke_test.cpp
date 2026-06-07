#include <catch2/catch_test_macros.hpp>

#include "neuronix/neuronix.hpp"

TEST_CASE("Neuronix exposes framework identity", "[smoke]") {
    REQUIRE(neuronix::framework_name() == "Neuronix");
    REQUIRE(neuronix::version_major == 0);
    REQUIRE(neuronix::version_minor == 1);
    REQUIRE(neuronix::version_patch == 0);
}
