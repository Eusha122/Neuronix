#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// Improved CNN for MNIST — 3x3 convs, BatchNorm, Dropout, CosineAnnealing.
//
// Input layout: (784, N) — each column is one flattened, normalized image
//
//  Conv(1,16,3,28,28,pad=1)  same-pad  → (16*28*28, N)   [160 params]
//  ReLU
//  MaxPool(2,16,28,28)                  → (16*14*14, N)
//  Conv(16,32,3,14,14,pad=1) same-pad  → (32*14*14, N)   [4640 params]
//  ReLU
//  MaxPool(2,32,14,14)                  → (32*7*7,  N) = (1568, N)
//  Flatten
//  Dense(1568,256)                                        [401664 params]
//  BatchNorm(256)                                         [512 params]
//  ReLU
//  Dropout(0.35)
//  Dense(256,10)                                          [2570 params]
//
//  Total: ~409546 params   Target: >99% test accuracy

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/mnist";

    std::cout << "Loading MNIST from: " << data_dir << '\n';

    neuronix::MnistDataset train, test;
    try {
        train = neuronix::MnistLoader::load_train(data_dir);
        test  = neuronix::MnistLoader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    std::cout << "Train: " << train.size() << "  Test: " << test.size() << '\n';

    neuronix::seed_random(42);

    neuronix::Model model;
    model.add<neuronix::Conv2D>(1, 16, 3, 28, 28, 1);   // same-pad
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 16, 28, 28);
    model.add<neuronix::Conv2D>(16, 32, 3, 14, 14, 1);  // same-pad
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 32, 14, 14);
    model.add<neuronix::Flatten>();
    model.add<neuronix::Dense>(1568, 256);
    model.add<neuronix::BatchNorm>(256);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dropout>(0.35);
    model.add<neuronix::Dense>(256, 10);
    model.summary();

    neuronix::CrossEntropyLoss                   loss_fn;
    neuronix::Adam                               opt{model, 0.001};
    neuronix::CosineAnnealingLR<neuronix::Adam>  sched{opt, 15, 0.00005};

    const std::size_t epochs     = 15;
    const std::size_t batch_size = 128;

    std::cout << "\nTraining: " << epochs << " epochs, batch=" << batch_size
              << ", Adam lr=1e-3 -> 5e-5 (cosine)\n";
    std::cout << std::string(70, '=') << '\n';

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train.images, train.labels_onehot,
                                    batch_size, true};
        double      total_loss = 0.0;
        std::size_t n_batches  = 0;

        for (auto [X, y] : loader) {
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

    model.save("lenet_mnist.nnx");
    std::cout << "\nModel saved to lenet_mnist.nnx\n";
    return 0;
}
