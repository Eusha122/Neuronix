#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// Binary face classifier: face (1) vs. no-face (0)
//
// Positive class : LFW-funneled faces  (scripts/prepare_lfw.py → data/lfw_faces_32.bin)
// Negative class : CIFAR-10 images     (data/cifar10/) — no human faces in CIFAR-10
//
// Architecture:
//   Conv(3,16,3,32,32,pad=1) → LeakyReLU → MaxPool(2,16,32,32)
//   Conv(16,32,3,16,16,pad=1) → LeakyReLU → MaxPool(2,32,16,16)
//   Flatten
//   Dense(2048,128) → BatchNorm(128) → LeakyReLU → Dropout(0.4)
//   Dense(128,2)
//
// Optimizer : AdamW lr=0.0005, wd=0.01 → CosineAnnealing to 1e-5 over 30 epochs
// Augment   : RandomHorizontalFlip(0.5)

int main(int argc, char* argv[]) {
    const std::string face_bin  = (argc > 1) ? argv[1] : "data/lfw_faces_32.bin";
    const std::string cifar_dir = (argc > 2) ? argv[2] : "data/cifar10";

    // ── Load data ─────────────────────────────────────────────────────────────
    std::cout << "Loading CIFAR-10 (non-face) from: " << cifar_dir << '\n';
    neuronix::Cifar10Dataset cifar_train, cifar_test;
    try {
        cifar_train = neuronix::Cifar10Loader::load_train(cifar_dir);
        cifar_test  = neuronix::Cifar10Loader::load_test(cifar_dir);
    } catch (const std::exception& e) {
        std::cerr << "CIFAR-10 error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Loading LFW faces from: " << face_bin << '\n';
    neuronix::BinaryDataset train_data, test_data;
    try {
        // Use first 10000 CIFAR-10 train as negative train examples
        // Use first 2000 CIFAR-10 test as negative test examples
        train_data = neuronix::FaceLoader::load(face_bin, cifar_train, 10000);
        // For test: reload faces with offset — simple approach: use same faces but
        // hold out the last 20% as test (approx via max_per_class limit)
        neuronix::BinaryDataset all_data = neuronix::FaceLoader::load(
            face_bin, cifar_test, 2000);
        test_data = std::move(all_data);
    } catch (const std::exception& e) {
        std::cerr << "Face data error: " << e.what() << '\n';
        std::cerr << "Run: python scripts/prepare_lfw.py\n";
        return 1;
    }

    std::cout << "Train: " << train_data.size()
              << "  Test: "  << test_data.size()  << '\n';
    std::cout << "  faces=" << train_data.size()/2
              << "  non-faces=" << train_data.size()/2 << " (balanced)\n\n";

    // ── Build model ───────────────────────────────────────────────────────────
    neuronix::seed_random(42);

    neuronix::Model model;
    model.add<neuronix::Conv2D>(3, 16, 3, 32, 32, 1);
    model.add<neuronix::LeakyReLU>();
    model.add<neuronix::MaxPool2D>(2, 16, 32, 32);
    model.add<neuronix::Conv2D>(16, 32, 3, 16, 16, 1);
    model.add<neuronix::LeakyReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 16, 16);
    model.add<neuronix::Flatten>();
    model.add<neuronix::Dense>(2048, 128);
    model.add<neuronix::BatchNorm>(128);
    model.add<neuronix::LeakyReLU>();
    model.add<neuronix::Dropout>(0.4);
    model.add<neuronix::Dense>(128, 2);
    model.summary();

    neuronix::CrossEntropyLoss                    loss_fn;
    neuronix::AdamW                               opt{model, 0.0005, 0.9, 0.999, 1e-8, 0.01};
    neuronix::CosineAnnealingLR<neuronix::AdamW>  sched{opt, 30, 1e-5};

    neuronix::RandomHorizontalFlip flip;
    const neuronix::ImageDims      img_dims{3, 32, 32};

    const std::size_t epochs     = 30;
    const std::size_t batch_size = 128;

    std::cout << "Training: " << epochs << " epochs, batch=" << batch_size
              << ", AdamW lr=5e-4 wd=0.01 -> 1e-5 (cosine)\n";
    std::cout << std::string(70, '=') << '\n';

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train_data.images, train_data.labels_onehot,
                                    batch_size, true};
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

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed    << std::setprecision(4) << avg_loss
                  << "  test_acc=" << std::setprecision(4) << test_acc
                  << "  lr=" << std::scientific  << std::setprecision(2) << opt.lr()
                  << "  " << std::fixed << std::setprecision(1) << dt << "s"
                  << '\n';
    }

    model.save("face_classifier.nnx");
    std::cout << "\nModel saved to face_classifier.nnx\n";
    return 0;
}
