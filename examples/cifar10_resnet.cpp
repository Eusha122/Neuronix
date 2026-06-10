#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// CIFAR-10 with residual blocks.
//
// Architecture (ResBlocks only at 8×8 — same cost as original CIFAR-10 + small extra):
//   Conv(3,32,3,32,32,1) → ReLU → MaxPool   → 32×16×16
//   Conv(32,32,3,16,16,1) → ReLU → MaxPool  → 32×8×8
//   ResBlock(32, 8, 8)     identity skip     (2 Conv2D at cheap 8×8)
//   ResBlock(32, 8, 8)     identity skip
//   Flatten
//   Dense(2048, 256) → BN(256) → ReLU → Dropout(0.3)
//   Dense(256, 10)
//
// Optimizer: AdamW lr=0.001 wd=0.01 → CosineAnnealing to 1e-5 over 40 epochs
// Augment:   RandomHorizontalFlip + RandomCrop(32, pad=4)

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/cifar10";

    // ── Load data ─────────────────────────────────────────────────────────────
    std::cout << "Loading CIFAR-10 from: " << data_dir << '\n';
    neuronix::Cifar10Dataset train, test;
    try {
        train = neuronix::Cifar10Loader::load_train(data_dir);
        test  = neuronix::Cifar10Loader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    std::cout << "Train: " << train.size() << "  Test: " << test.size() << '\n';

    // ── Build model ───────────────────────────────────────────────────────────
    neuronix::seed_random(42);

    neuronix::Model model;
    // Downsample with plain convs first (matches proven CIFAR-10 speed)
    model.add<neuronix::Conv2D>(3, 32, 3, 32, 32, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 32, 32);
    model.add<neuronix::Conv2D>(32, 32, 3, 16, 16, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 16, 16);
    // ResBlocks only at 8×8 — cheap spatial size
    model.add<neuronix::ResidualBlock>(32, 8, 8);
    model.add<neuronix::ResidualBlock>(32, 8, 8);
    // Classifier
    model.add<neuronix::Flatten>();
    model.add<neuronix::Dense>(2048, 256);
    model.add<neuronix::BatchNorm>(256);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dropout>(0.3);
    model.add<neuronix::Dense>(256, 10);
    model.summary();

    neuronix::CrossEntropyLoss                   loss_fn;
    neuronix::AdamW                              opt{model, 0.001, 0.9, 0.999, 1e-8, 0.01};
    neuronix::CosineAnnealingLR<neuronix::AdamW> sched{opt, 40, 1e-5};

    neuronix::RandomHorizontalFlip flip;
    neuronix::RandomCrop           crop{32, 32, 4};
    const neuronix::ImageDims      img_dims{3, 32, 32};

    const std::size_t epochs     = 40;
    const std::size_t batch_size = 128;

    std::cout << "Training: " << epochs << " epochs, batch=" << batch_size
              << ", AdamW lr=1e-3 wd=0.01 -> 1e-5 (cosine)\n";
    std::cout << std::string(72, '=') << '\n' << std::flush;

    double best_acc = 0.0;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train.images, train.labels_onehot,
                                    batch_size, /*shuffle=*/true};
        double      total_loss = 0.0;
        std::size_t n_batches  = 0;

        for (auto [X, y] : loader) {
            X = flip.apply(X, img_dims);
            X = crop.apply(X, img_dims.C);
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
            model.predict(test.images), test.labels);

        if (test_acc > best_acc) {
            best_acc = test_acc;
            model.save("cifar10_resnet.nnx");
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
    std::cout << "Model saved to cifar10_resnet.nnx\n";
    return 0;
}
