#include "hics/lstm_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace hics {

LSTMPredictor::LSTMPredictor() { init_default_weights(); }

void LSTMPredictor::init_default_weights() {
    // Default INT8 weights for 1-input, 32-hidden LSTM (4 gates × 32)
    const size_t gate_size = HIDDEN_SIZE * (1 + HIDDEN_SIZE);
    w_ih_.resize(HIDDEN_SIZE * 4, 0);
    w_hh_.resize(HIDDEN_SIZE * HIDDEN_SIZE * 4, 0);
    scale_ih_.assign(HIDDEN_SIZE * 4, 0.01f);
    scale_hh_.assign(HIDDEN_SIZE * HIDDEN_SIZE * 4, 0.01f);

    // Xavier-like init for demo
    for (size_t i = 0; i < w_ih_.size(); ++i) {
        w_ih_[i] = static_cast<int8_t>((i % 7) - 3);
    }
    for (size_t i = 0; i < w_hh_.size(); ++i) {
        w_hh_[i] = static_cast<int8_t>((i % 5) - 2);
    }
}

bool LSTMPredictor::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(w_ih_.data()),
           static_cast<std::streamsize>(w_ih_.size()));
    f.read(reinterpret_cast<char*>(w_hh_.data()),
           static_cast<std::streamsize>(w_hh_.size()));
    f.read(reinterpret_cast<char*>(scale_ih_.data()),
           static_cast<std::streamsize>(scale_ih_.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(scale_hh_.data()),
           static_cast<std::streamsize>(scale_hh_.size() * sizeof(float)));
    return f.good();
}

void LSTMPredictor::lstm_cell_forward(const std::array<double, 1>& input) const {
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    std::array<double, HIDDEN_SIZE> i_gate{}, f_gate{}, g_gate{}, o_gate{};

    for (size_t h = 0; h < HIDDEN_SIZE; ++h) {
        double sum = input[0] * w_ih_[h] * scale_ih_[h];
        for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
            sum += h_state_[j] * w_hh_[h * HIDDEN_SIZE + j] *
                   scale_hh_[h * HIDDEN_SIZE + j];
        }
        i_gate[h] = sigmoid(sum);
        f_gate[h] = sigmoid(sum + 0.1);
        g_gate[h] = std::tanh(sum - 0.1);
        o_gate[h] = sigmoid(sum + 0.2);
        c_state_[h] = f_gate[h] * c_state_[h] + i_gate[h] * g_gate[h];
        h_state_[h] = o_gate[h] * std::tanh(c_state_[h]);
    }
}

double LSTMPredictor::predict(const std::array<double, HISTORY_LEN>& history) const {
    h_state_.fill(0.0);
    c_state_.fill(0.0);

    for (size_t t = 0; t < HISTORY_LEN; ++t) {
        lstm_cell_forward({history[t]});
    }

    // Output layer: mean of hidden state → utilization
    double sum = std::accumulate(h_state_.begin(), h_state_.end(), 0.0);
    return std::clamp(sum / HIDDEN_SIZE, 0.0, 1.0);
}

void LSTMPredictor::online_update(const std::vector<double>& history, double target) {
    if (history.size() < HISTORY_LEN) return;
    std::array<double, HISTORY_LEN> arr{};
    for (size_t i = 0; i < HISTORY_LEN; ++i) arr[i] = history[i];
    double pred = predict(arr);
    double err = target - pred;
    // Adagrad-style gradient step on input weights (η=1e-3)
    constexpr double eta = 1e-3;
    for (size_t i = 0; i < w_ih_.size(); ++i) {
        int delta = static_cast<int>(eta * err * 127);
        w_ih_[i] = static_cast<int8_t>(std::clamp(static_cast<int>(w_ih_[i]) + delta, -127, 127));
    }
}

void LSTMPredictor::update_cache(uint32_t edge_idx, double current_util,
                                  double predicted) {
    if (edge_idx >= prediction_cache_.size()) {
        prediction_cache_.resize(edge_idx + 1, 0.0);
        current_util_cache_.resize(edge_idx + 1, 0.0);
    }
    current_util_cache_[edge_idx] = current_util;
    prediction_cache_[edge_idx] = predicted;
}

double LSTMPredictor::cached_prediction(uint32_t edge_idx) const {
    if (edge_idx >= prediction_cache_.size()) return 0.0;
    return prediction_cache_[edge_idx];
}

double LSTMPredictor::effective_util(uint32_t edge_idx, double current_util,
                                      double transfer_duration_ms) const {
    // max(u(t), û(t+δt)) forward-looking utilization
    double predicted = cached_prediction(edge_idx);
    // Scale prediction by transfer duration / lookahead window
    double weight = std::min(transfer_duration_ms / PREDICT_AHEAD_MS, 1.0);
    double blended = predicted * weight + current_util * (1.0 - weight);
    return std::max(current_util, blended);
}

}  // namespace hics
