#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "neuronix/neuronix.hpp"

// Load a flat pixel binary written by scripts/infer.py:
//   [4 bytes LE uint32: num_pixels]
//   [num_pixels * 8 bytes: float64 pixels, already normalised]
static neuronix::Matrix load_pixels(const std::string& path) {
    std::ifstream f{path, std::ios::binary};
    if (!f) throw std::runtime_error{"Cannot open pixel file: " + path};

    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);

    neuronix::Matrix img = neuronix::Matrix::zeros(n, 1);
    for (std::uint32_t i = 0; i < n; ++i) {
        double v = 0.0;
        f.read(reinterpret_cast<char*>(&v), 8);
        img(i, 0) = v;
    }
    return img;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: neuronix_infer <model.nnx> <pixels.bin>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string pixel_path = argv[2];

    neuronix::Matrix input;
    try {
        input = load_pixels(pixel_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    neuronix::Model model;
    try {
        model = neuronix::Model::load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << '\n';
        return 1;
    }

    // Inference forward pass (Dropout disabled)
    neuronix::Matrix logits = model.predict(input);
    const std::size_t num_classes = logits.rows();

    // Stable softmax
    double max_v = logits(0, 0);
    for (std::size_t i = 1; i < num_classes; ++i)
        max_v = std::max(max_v, logits(i, 0));

    std::vector<double> probs(num_classes);
    double sum = 0.0;
    for (std::size_t i = 0; i < num_classes; ++i) {
        probs[i] = std::exp(logits(i, 0) - max_v);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Output: num_classes, then one probability per line
    std::cout << num_classes << '\n';
    for (double p : probs)
        std::cout << std::fixed << std::setprecision(8) << p << '\n';

    return 0;
}
