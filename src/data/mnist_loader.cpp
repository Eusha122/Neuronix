#include "neuronix/data/mnist_loader.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace neuronix {

// IDX files are big-endian; Windows is little-endian — swap bytes.
std::uint32_t MnistLoader::read_u32_be(std::istream& s) {
    std::uint8_t b[4];
    s.read(reinterpret_cast<char*>(b), 4);
    return (std::uint32_t{b[0]} << 24) | (std::uint32_t{b[1]} << 16)
         | (std::uint32_t{b[2]} <<  8) |  std::uint32_t{b[3]};
}

MnistDataset MnistLoader::load(const std::string& img_path,
                                const std::string& lbl_path) {
    // ── Images ──────────────────────────────────────────────────────────────
    std::ifstream img_f{img_path, std::ios::binary};
    if (!img_f) throw std::runtime_error{"MnistLoader: cannot open " + img_path};

    if (read_u32_be(img_f) != 0x00000803)
        throw std::runtime_error{"MnistLoader: bad image file magic"};

    const std::uint32_t num    = read_u32_be(img_f);
    const std::uint32_t rows   = read_u32_be(img_f);
    const std::uint32_t cols   = read_u32_be(img_f);
    const std::size_t   pixels = rows * cols;

    // images stored as (784, N) — each column is one flattened image
    Matrix images = Matrix::zeros(pixels, num);
    for (std::uint32_t n = 0; n < num; ++n) {
        for (std::size_t p = 0; p < pixels; ++p) {
            std::uint8_t byte;
            img_f.read(reinterpret_cast<char*>(&byte), 1);
            images(p, n) = static_cast<double>(byte) / 255.0;
        }
    }

    // ── Labels ───────────────────────────────────────────────────────────────
    std::ifstream lbl_f{lbl_path, std::ios::binary};
    if (!lbl_f) throw std::runtime_error{"MnistLoader: cannot open " + lbl_path};

    if (read_u32_be(lbl_f) != 0x00000801)
        throw std::runtime_error{"MnistLoader: bad label file magic"};
    if (read_u32_be(lbl_f) != num)
        throw std::runtime_error{"MnistLoader: image/label count mismatch"};

    std::vector<int> labels(num);
    Matrix onehot = Matrix::zeros(10, num);
    for (std::uint32_t n = 0; n < num; ++n) {
        std::uint8_t byte;
        lbl_f.read(reinterpret_cast<char*>(&byte), 1);
        labels[n]           = static_cast<int>(byte);
        onehot(byte, n)     = 1.0;
    }

    return MnistDataset{std::move(images), std::move(onehot), std::move(labels)};
}

MnistDataset MnistLoader::load_train(const std::string& data_dir) {
    return load(data_dir + "/train-images-idx3-ubyte",
                data_dir + "/train-labels-idx1-ubyte");
}

MnistDataset MnistLoader::load_test(const std::string& data_dir) {
    return load(data_dir + "/t10k-images-idx3-ubyte",
                data_dir + "/t10k-labels-idx1-ubyte");
}

} // namespace neuronix
