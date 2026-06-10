#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "neuronix/neuronix.hpp"

// Sentiment analysis on IMDB movie reviews (positive / negative).
//
// Dataset: 25,000 train / 25,000 test reviews, binary labels
// Prepare: python scripts/prepare_imdb.py
//
// Architecture:
//   Embedding(10000, 64, seq_len=100)   → (6400, N)
//   LSTM(input=64, hidden=64, seq=100)  → (64, N)   last hidden state
//   Dense(64, 2)                        → (2, N)
//
// Optimizer: Adam lr=0.001 → CosineAnnealing to 1e-4 over 15 epochs

int main(int argc, char* argv[]) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data/imdb";

    // ── Load data ─────────────────────────────────────────────────────────────
    std::cout << "Loading IMDB from: " << data_dir << '\n';
    neuronix::ImdbDataset train_data, test_data;
    try {
        train_data = neuronix::ImdbLoader::load_train(data_dir);
        test_data  = neuronix::ImdbLoader::load_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Run: python scripts/prepare_imdb.py\n";
        return 1;
    }

    std::cout << "Train: " << train_data.size()
              << "   Test: " << test_data.size()
              << "   Seq len: " << train_data.seq_len() << '\n';
    std::cout << "  Classes: negative (0) / positive (1)\n\n";

    // ── Build model ───────────────────────────────────────────────────────────
    neuronix::seed_random(42);

    constexpr std::size_t vocab_size = 10000;
    constexpr std::size_t embed_dim  = 64;
    constexpr std::size_t hidden     = 64;
    const     std::size_t seq_len    = train_data.seq_len();

    neuronix::Model model;
    model.add<neuronix::Embedding>(vocab_size, embed_dim, seq_len);
    model.add<neuronix::LSTM>(embed_dim, hidden, seq_len);
    model.add<neuronix::Dense>(hidden, 2);
    model.summary();

    neuronix::CrossEntropyLoss                  loss_fn;
    neuronix::Adam                              opt{model, 0.001};
    neuronix::CosineAnnealingLR<neuronix::Adam> sched{opt, 15, 1e-4};

    const std::size_t epochs     = 15;
    const std::size_t batch_size = 128;

    std::cout << "Training: " << epochs << " epochs"
              << ", batch=" << batch_size
              << ", Adam lr=1e-3 -> 1e-4 (cosine)\n";
    std::cout << std::string(68, '=') << '\n' << std::flush;

    double best_acc = 0.0;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();

        neuronix::DataLoader loader{train_data.sequences, train_data.labels_onehot,
                                    batch_size, /*shuffle=*/true};
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
            model.predict(test_data.sequences), test_data.labels);

        if (test_acc > best_acc) {
            best_acc = test_acc;
            model.save("sentiment.nnx");
        }

        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << "  loss=" << std::fixed    << std::setprecision(4) << avg_loss
                  << "  test_acc=" << std::setprecision(4) << test_acc
                  << (test_acc >= best_acc ? " *" : "  ")
                  << "  lr=" << std::scientific  << std::setprecision(2) << opt.lr()
                  << "  " << std::fixed << std::setprecision(1) << dt << "s"
                  << '\n' << std::flush;
    }

    std::cout << "\nBest accuracy: " << std::fixed << std::setprecision(4)
              << best_acc << '\n';
    std::cout << "Model saved to sentiment.nnx\n";
    return 0;
}
