#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// Handwritten letter recognition on EMNIST-Letters (A-Z, 26 classes)
//
// Dataset: 124,800 train / 20,800 test images, 28x28 grayscale
// Prepare:  python scripts/prepare_emnist.py
//
// Architecture (matches face classifier pattern — no post-conv BN for speed):
//   Conv(1,16,3,28,28,pad=1) → ReLU → MaxPool(2,16,28,28)    → 16×14×14
//   Conv(16,32,3,14,14,pad=1) → ReLU → MaxPool(2,32,14,14)   → 32×7×7 = 1568
//   Flatten
//   Dense(1568,256) → BN(256) → ReLU → Dropout(0.4)
//   Dense(256,26)
//
// Optimizer: AdamW lr=0.001 wd=0.01 → CosineAnnealing to 1e-4 over 20 epochs

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/emnist";

    // ── Load data ─────────────────────────────────────────────────────────────
    std::cout << "Loading EMNIST-Letters from: " << data_dir << '\n';
    neuronix::EmnistDataset train_data, test_data;
    try {
        train_data = neuronix::EmnistLoader::load_train(data_dir);
        test_data  = neuronix::EmnistLoader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Run: python scripts/prepare_emnist.py\n";
        return 1;
    }

    std::cout << "Train: " << train_data.size()
              << "   Test: " << test_data.size()
              << "   Classes: 26 (A-Z)\n\n";

    // Per-channel (grayscale) mean/std normalisation
    const std::vector<double> mean = {0.1722};
    const std::vector<double> std  = {0.3309};
    train_data.images = neuronix::normalize(train_data.images, mean, std,
                                            /*C=*/1, /*H=*/28, /*W=*/28);
    test_data.images  = neuronix::normalize(test_data.images,  mean, std,
                                            /*C=*/1, /*H=*/28, /*W=*/28);

    // ── Build model ───────────────────────────────────────────────────────────
    neuronix::seed_random(42);

    neuronix::Model model;
    // Block 1 — 28×28 → 14×14
    model.add<neuronix::Conv2D>(1, 16, 3, 28, 28, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 16, 28, 28);
    // Block 2 — 14×14 → 7×7
    model.add<neuronix::Conv2D>(16, 32, 3, 14, 14, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 14, 14);
    // Classifier
    model.add<neuronix::Flatten>();
    model.add<neuronix::Dense>(1568, 256);
    model.add<neuronix::BatchNorm>(256);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dropout>(0.4);
    model.add<neuronix::Dense>(256, 26);
    model.summary();

    neuronix::CrossEntropyLoss                   loss_fn;
    neuronix::AdamW                              opt{model, 0.001, 0.9, 0.999, 1e-8, 0.01};
    neuronix::CosineAnnealingLR<neuronix::AdamW> sched{opt, 20, 1e-4};

    neuronix::RandomHorizontalFlip flip;
    const neuronix::ImageDims      img_dims{1, 28, 28};

    const std::size_t epochs     = 20;
    const std::size_t batch_size = 128;

    std::cout << "Training: " << epochs << " epochs, batch=" << batch_size
              << ", AdamW lr=1e-3 wd=0.01 -> 1e-4 (cosine)\n"
              << "Expected: ~4-5 min/epoch on Ryzen 7 5700G (~90 min total)\n";
    std::cout << std::string(72, '=') << '\n';
    std::cout << std::flush;

    double best_acc = 0.0;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train_data.images, train_data.labels_onehot,
                                    batch_size, /*shuffle=*/true};
        double      total_loss = 0.0;
        std::size_t n_batches  = 0;

        for (auto [X, y] : loader) {
            X = flip.apply(X, img_dims);
            opt.zero_grad();
            neuronix::Matrix out = model.train_forward(X);
            total_loss += loss_fn.forward(out, y);
            model.backward(loss_fn.backward());
            opt.step();
            ++n_batches;
        }

        sched.step();

        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double avg_loss = total_loss / static_cast<double>(n_batches);
        const double test_acc = neuronix::metrics::accuracy(
            model.predict(test_data.images), test_data.labels);

        if (test_acc > best_acc) {
            best_acc = test_acc;
            model.save("emnist_letters.nnx");
        }

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed    << std::setprecision(4) << avg_loss
                  << "  test_acc=" << std::setprecision(4) << test_acc
                  << (test_acc >= best_acc ? " *" : "  ")
                  << "  lr=" << std::scientific  << std::setprecision(2) << opt.lr()
                  << "  " << std::fixed << std::setprecision(1) << dt << "s"
                  << '\n' << std::flush;
    }

    std::cout << "\nBest test accuracy: " << std::fixed << std::setprecision(4)
              << best_acc << '\n';
    std::cout << "Model saved to emnist_letters.nnx\n";
    return 0;
}
