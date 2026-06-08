#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// Lean 2-block CNN for CIFAR-10 — CPU-friendly (single conv per resolution).
//
// Input: (3*32*32=3072, N)  — pixels normalised to [0,1], channel-first
//
// Block 1:
//   Conv(3,16,3,32,32,pad=1) same-pad → (16*32*32, N)   [448 params]
//   ReLU → MaxPool(2,16,32,32)         → (16*16*16=4096, N)
//
// Block 2:
//   Conv(16,32,3,16,16,pad=1) same-pad → (32*16*16, N)  [4640 params]
//   ReLU → MaxPool(2,32,16,16)          → (32*8*8=2048, N)
//
// Classifier:
//   Flatten
//   Dense(2048,256) → BatchNorm(256) → ReLU → Dropout(0.4)
//   Dense(256,10)
//
// Double-conv blocks removed: im2col matrix shrinks from 300 MB → 37 MB/batch.
// Augmentation: RandomHorizontalFlip(0.5) + RandomCrop(32,32,padding=4)
// Optimizer:    Adam lr=0.001 → CosineAnnealing to 1e-5 over 50 epochs

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/cifar10";

    std::cout << "Loading CIFAR-10 from: " << data_dir << '\n';

    neuronix::Cifar10Dataset train, test;
    try {
        train = neuronix::Cifar10Loader::load_train(data_dir);
        test  = neuronix::Cifar10Loader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Download CIFAR-10 binary and extract to: " << data_dir << '\n';
        std::cerr << "  data_batch_1.bin .. data_batch_5.bin  test_batch.bin\n";
        return 1;
    }
    std::cout << "Train: " << train.size() << "  Test: " << test.size() << '\n';

    neuronix::seed_random(42);

    neuronix::Model model;
    // Block 1
    model.add<neuronix::Conv2D>(3, 16, 3, 32, 32, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 16, 32, 32);
    // Block 2
    model.add<neuronix::Conv2D>(16, 32, 3, 16, 16, 1);
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 16, 16);
    // Classifier
    model.add<neuronix::Flatten>();
    model.add<neuronix::Dense>(2048, 256);
    model.add<neuronix::BatchNorm>(256);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dropout>(0.4);
    model.add<neuronix::Dense>(256, 10);
    model.summary();

    neuronix::CrossEntropyLoss                    loss_fn;
    neuronix::Adam                                opt{model, 0.001};
    neuronix::CosineAnnealingLR<neuronix::Adam>   sched{opt, 50, 1e-5};

    neuronix::RandomHorizontalFlip flip;
    neuronix::RandomCrop           crop{32, 32, 4};
    const neuronix::ImageDims      img_dims{3, 32, 32};

    const std::size_t epochs     = 50;
    const std::size_t batch_size = 128;

    std::cout << "\nTraining: " << epochs << " epochs, batch=" << batch_size
              << ", Adam lr=1e-3 -> 1e-5 (cosine)\n";
    std::cout << "Augmentation: RandomHorizontalFlip(0.5) + RandomCrop(pad=4)\n";
    std::cout << std::string(72, '=') << '\n';

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train.images, train.labels_onehot,
                                    batch_size, true};
        double      total_loss = 0.0;
        std::size_t n_batches  = 0;

        for (auto [X, y] : loader) {
            // Apply augmentation to each training batch
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

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed    << std::setprecision(4) << avg_loss
                  << "  test_acc=" << std::setprecision(4) << test_acc
                  << "  lr=" << std::scientific  << std::setprecision(2) << opt.lr()
                  << "  " << std::fixed << std::setprecision(1) << dt << "s"
                  << '\n';
    }

    model.save("cifar10_vgg.nnx");
    std::cout << "\nModel saved to cifar10_vgg.nnx\n";
    return 0;
}
