#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// LeNet-5 inspired architecture for MNIST.
//
// Input layout: (784, N) — same format as neuronix_mnist.exe
//
//  Conv(1,6,5,28,28,pad=2) → (6*28*28, N)   [156 params]
//  ReLU
//  MaxPool(2,6,28,28)      → (6*14*14, N)
//  Conv(6,16,5,14,14)      → (16*10*10,N)   [2416 params]
//  ReLU
//  MaxPool(2,16,10,10)     → (16*5*5,  N)   [= 400 features]
//  Dense(400,120)                            [48120 params]
//  ReLU
//  Dense(120,84)                             [10164 params]
//  ReLU
//  Dense(84,10)                              [850 params]
//
//  Total: 61706 params   (vs 235146 for the fully-connected baseline)

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
    model.add<neuronix::Conv2D>(1, 6, 5, 28, 28, 2);   // same-pad
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 6, 28, 28);
    model.add<neuronix::Conv2D>(6, 16, 5, 14, 14);     // valid-pad
    model.add<neuronix::ReLU>();
    model.add<neuronix::MaxPool2D>(2, 16, 10, 10);
    model.add<neuronix::Dense>(400, 120);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dense>(120, 84);
    model.add<neuronix::ReLU>();
    model.add<neuronix::Dense>(84, 10);
    model.summary();

    neuronix::CrossEntropyLoss loss_fn;
    neuronix::Adam             opt{model, 0.001};

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
        const double test_acc  = neuronix::metrics::accuracy(
            model.predict(test.images), test.labels);

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  test_acc=" << std::setprecision(4) << test_acc << '\n';
    }

    model.save("lenet_mnist.nnx");
    std::cout << "Model saved to lenet_mnist.nnx\n";
    return 0;
}
