#include <iostream>

#include "neuronix/neuronix.hpp"

int main() {
    std::cout << neuronix::framework_name() << " "
              << neuronix::version_major << "."
              << neuronix::version_minor << "."
              << neuronix::version_patch << '\n';

    return 0;
}
