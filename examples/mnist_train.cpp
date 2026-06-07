#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"
#include "neuronix/data/mnist_loader.hpp"
#include "neuronix/data/data_loader.hpp"
#include "neuronix/metrics/accuracy.hpp"

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/mnist";

    std::cout << "Loading MNIST from: " << data_dir << '\n';

    neuronix::MnistDataset train, test;
    try {
        train = neuronix::MnistLoader::load_train(data_dir);
        test  = neuronix::MnistLoader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Download MNIST files to: " << data_dir << '\n';
        std::cerr << "  train-images-idx3-ubyte\n";
        std::cerr << "  train-labels-idx1-ubyte\n";
        std::cerr << "  t10k-images-idx3-ubyte\n";
        std::cerr << "  t10k-labels-idx1-ubyte\n";
        return 1;
    }

    std::cout << "Train samples: " << train.size()
              << "  Test samples: " << test.size() << '\n';

    // ── Build model ──────────────────────────────────────────────────────────
    neuronix::seed_random(42);

    neuronix::Model model;
    model.add<neuronix::Dense>(784, 256);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dense>(256, 128);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dense>(128, 10);
    model.summary();

    // ── Train ────────────────────────────────────────────────────────────────
    neuronix::CrossEntropyLoss loss_fn;
    neuronix::SGD              opt{model, 0.01};

    const std::size_t epochs     = 10;
    const std::size_t batch_size = 64;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        neuronix::DataLoader loader{train.images, train.labels_onehot,
                                    batch_size, true};
        double total_loss = 0.0;

        for (auto [X, y] : loader) {
            opt.zero_grad();
            neuronix::Matrix out = model.train_forward(X);
            total_loss += loss_fn.forward(out, y);
            model.backward(loss_fn.backward());
            opt.step();
        }

        loader.reshuffle();

        const double avg_loss  = total_loss / static_cast<double>(loader.num_batches());
        const double train_acc = neuronix::metrics::accuracy(
            model.predict(train.images), train.labels);
        const double test_acc  = neuronix::metrics::accuracy(
            model.predict(test.images),  test.labels);

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  train_acc=" << std::setprecision(4) << train_acc
                  << "  test_acc="  << std::setprecision(4) << test_acc
                  << '\n';
    }

    return 0;
}
