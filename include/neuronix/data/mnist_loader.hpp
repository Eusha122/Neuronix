#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "neuronix/core/matrix.hpp"

namespace neuronix {

struct MnistDataset {
    Matrix images;              // (784, N) — flattened, normalised to [0, 1]
    Matrix labels_onehot;       // (10, N)  — one-hot encoded
    std::vector<int> labels;    // N        — raw class indices 0-9

    [[nodiscard]] std::size_t size() const noexcept { return labels.size(); }
};

class MnistLoader {
public:
    // data_dir must contain the four uncompressed IDX files:
    //   train-images-idx3-ubyte   train-labels-idx1-ubyte
    //   t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
    [[nodiscard]] static MnistDataset load_train(const std::string& data_dir);
    [[nodiscard]] static MnistDataset load_test (const std::string& data_dir);

private:
    [[nodiscard]] static Matrix           load_images(const std::string& path);
    [[nodiscard]] static MnistDataset     load(const std::string& images_path,
                                               const std::string& labels_path);
    [[nodiscard]] static std::uint32_t    read_u32_be(std::istream& s);
};

} // namespace neuronix
