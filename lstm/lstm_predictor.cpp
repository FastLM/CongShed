#include "lstm/lstm_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace hics {
namespace {

inline double sigmoid(double x) {
    x = std::clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
bool read_pod(std::ifstream& f, T& out) {
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(f);
}

template <typename T>
bool write_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return static_cast<bool>(f);
}

}  // namespace

LSTMPredictor::LSTMPredictor() { init_default_weights(); }

void LSTMPredictor::init_default_weights() {
    layers_.resize(NUM_LAYERS);
    for (size_t L = 0; L < NUM_LAYERS; ++L) {
        auto& layer = layers_[L];
        layer.input_size = (L == 0) ? 1u : static_cast<uint32_t>(HIDDEN_SIZE);
        const size_t rows = GATES * HIDDEN_SIZE;
        const size_t cols_ih = layer.input_size;
        layer.w_ih.assign(rows * cols_ih, 0);
        layer.w_hh.assign(rows * HIDDEN_SIZE, 0);
        layer.b_ih.assign(rows, 0.0f);
        layer.b_hh.assign(rows, 0.0f);
        layer.scale_ih = 0.02f;
        layer.scale_hh = 0.02f;

        // Deterministic Xavier-like INT8 init for demos without a weight file
        for (size_t i = 0; i < layer.w_ih.size(); ++i) {
            layer.w_ih[i] = static_cast<int8_t>(((i * 17 + L * 3) % 11) - 5);
        }
        for (size_t i = 0; i < layer.w_hh.size(); ++i) {
            layer.w_hh[i] = static_cast<int8_t>(((i * 13 + L * 7) % 9) - 4);
        }
        // Forget-gate bias warm-start (~1.0 in float → help retain history)
        for (size_t h = 0; h < HIDDEN_SIZE; ++h) {
            layer.b_ih[HIDDEN_SIZE + h] = 1.0f;
            layer.b_hh[HIDDEN_SIZE + h] = 0.0f;
        }
    }

    head_w_.assign(HIDDEN_SIZE, 0);
    for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
        head_w_[i] = static_cast<int8_t>((static_cast<int>(i) % 7) - 3);
    }
    head_scale_ = 0.05f;
    head_b_ = 0.0f;

    ada_head_.assign(HIDDEN_SIZE, 0.0f);
    ada_w_ih0_.assign(layers_[0].w_ih.size(), 0.0f);
    weights_loaded_ = false;
}

bool LSTMPredictor::save_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f.write(LSTM_WEIGHT_MAGIC, 8);
    write_pod(f, LSTM_WEIGHT_VERSION);
    const uint32_t hidden = static_cast<uint32_t>(HIDDEN_SIZE);
    const uint32_t layers = static_cast<uint32_t>(NUM_LAYERS);
    const uint32_t history = static_cast<uint32_t>(HISTORY_LEN);
    write_pod(f, hidden);
    write_pod(f, layers);
    write_pod(f, history);

    for (const auto& layer : layers_) {
        write_pod(f, layer.input_size);
        const uint32_t rows = static_cast<uint32_t>(GATES * HIDDEN_SIZE);
        write_pod(f, rows);
        f.write(reinterpret_cast<const char*>(layer.w_ih.data()),
                static_cast<std::streamsize>(layer.w_ih.size()));
        write_pod(f, layer.scale_ih);
        f.write(reinterpret_cast<const char*>(layer.w_hh.data()),
                static_cast<std::streamsize>(layer.w_hh.size()));
        write_pod(f, layer.scale_hh);
        f.write(reinterpret_cast<const char*>(layer.b_ih.data()),
                static_cast<std::streamsize>(layer.b_ih.size() * sizeof(float)));
        f.write(reinterpret_cast<const char*>(layer.b_hh.data()),
                static_cast<std::streamsize>(layer.b_hh.size() * sizeof(float)));
    }

    const uint32_t head_n = static_cast<uint32_t>(HIDDEN_SIZE);
    write_pod(f, head_n);
    f.write(reinterpret_cast<const char*>(head_w_.data()),
            static_cast<std::streamsize>(head_w_.size()));
    write_pod(f, head_scale_);
    write_pod(f, head_b_);
    return static_cast<bool>(f);
}

bool LSTMPredictor::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[8]{};
    f.read(magic, 8);
    if (std::memcmp(magic, LSTM_WEIGHT_MAGIC, 8) != 0) {
        // Legacy v1: raw w_ih + w_hh + scales for layer-0 only (Python old export)
        f.clear();
        f.seekg(0);
        auto& layer = layers_[0];
        f.read(reinterpret_cast<char*>(layer.w_ih.data()),
               static_cast<std::streamsize>(layer.w_ih.size()));
        f.read(reinterpret_cast<char*>(layer.w_hh.data()),
               static_cast<std::streamsize>(layer.w_hh.size()));
        float s_ih = 0.01f, s_hh = 0.01f;
        f.read(reinterpret_cast<char*>(&s_ih), sizeof(float));
        f.read(reinterpret_cast<char*>(&s_hh), sizeof(float));
        if (!f) return false;
        layer.scale_ih = s_ih;
        layer.scale_hh = s_hh;
        weights_loaded_ = true;
        return true;
    }

    uint32_t version = 0, hidden = 0, nlayers = 0, history = 0;
    if (!read_pod(f, version) || !read_pod(f, hidden) || !read_pod(f, nlayers) ||
        !read_pod(f, history)) {
        return false;
    }
    if (version != LSTM_WEIGHT_VERSION || hidden != HIDDEN_SIZE ||
        nlayers != NUM_LAYERS) {
        return false;
    }

    layers_.assign(NUM_LAYERS, {});
    for (size_t L = 0; L < NUM_LAYERS; ++L) {
        auto& layer = layers_[L];
        uint32_t input_size = 0, rows = 0;
        if (!read_pod(f, input_size) || !read_pod(f, rows)) return false;
        layer.input_size = input_size;
        layer.w_ih.resize(static_cast<size_t>(rows) * input_size);
        layer.w_hh.resize(static_cast<size_t>(rows) * HIDDEN_SIZE);
        layer.b_ih.resize(rows);
        layer.b_hh.resize(rows);
        f.read(reinterpret_cast<char*>(layer.w_ih.data()),
               static_cast<std::streamsize>(layer.w_ih.size()));
        if (!read_pod(f, layer.scale_ih)) return false;
        f.read(reinterpret_cast<char*>(layer.w_hh.data()),
               static_cast<std::streamsize>(layer.w_hh.size()));
        if (!read_pod(f, layer.scale_hh)) return false;
        f.read(reinterpret_cast<char*>(layer.b_ih.data()),
               static_cast<std::streamsize>(layer.b_ih.size() * sizeof(float)));
        f.read(reinterpret_cast<char*>(layer.b_hh.data()),
               static_cast<std::streamsize>(layer.b_hh.size() * sizeof(float)));
        if (!f) return false;
    }

    uint32_t head_n = 0;
    if (!read_pod(f, head_n) || head_n != HIDDEN_SIZE) return false;
    head_w_.resize(HIDDEN_SIZE);
    f.read(reinterpret_cast<char*>(head_w_.data()),
           static_cast<std::streamsize>(head_w_.size()));
    if (!read_pod(f, head_scale_) || !read_pod(f, head_b_)) return false;

    ada_head_.assign(HIDDEN_SIZE, 0.0f);
    ada_w_ih0_.assign(layers_[0].w_ih.size(), 0.0f);
    weights_loaded_ = true;
    return true;
}

void LSTMPredictor::lstm_layer_forward(size_t layer_idx, const std::vector<double>& x,
                                       std::array<double, HIDDEN_SIZE>& h,
                                       std::array<double, HIDDEN_SIZE>& c) const {
    const auto& layer = layers_[layer_idx];
    const size_t rows = GATES * HIDDEN_SIZE;
    const size_t in_sz = layer.input_size;

    std::array<double, GATES * HIDDEN_SIZE> gates{};
    for (size_t r = 0; r < rows; ++r) {
        double sum = layer.b_ih[r] + layer.b_hh[r];
        for (size_t i = 0; i < in_sz; ++i) {
            sum += x[i] * (layer.w_ih[r * in_sz + i] * layer.scale_ih);
        }
        for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
            sum += h[j] * (layer.w_hh[r * HIDDEN_SIZE + j] * layer.scale_hh);
        }
        gates[r] = sum;
    }

    std::array<double, HIDDEN_SIZE> h_new{};
    for (size_t hid = 0; hid < HIDDEN_SIZE; ++hid) {
        const double i_g = sigmoid(gates[hid]);
        const double f_g = sigmoid(gates[HIDDEN_SIZE + hid]);
        const double g_g = std::tanh(gates[2 * HIDDEN_SIZE + hid]);
        const double o_g = sigmoid(gates[3 * HIDDEN_SIZE + hid]);
        c[hid] = f_g * c[hid] + i_g * g_g;
        h_new[hid] = o_g * std::tanh(c[hid]);
    }
    h = h_new;
}

double LSTMPredictor::head_forward(const std::array<double, HIDDEN_SIZE>& h) const {
    double sum = head_b_;
    for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
        sum += h[i] * (head_w_[i] * head_scale_);
    }
    return sigmoid(sum);
}

double LSTMPredictor::predict(const std::array<double, HISTORY_LEN>& history) const {
    std::array<double, HIDDEN_SIZE> h0{}, c0{}, h1{}, c1{};
    h0.fill(0.0);
    c0.fill(0.0);
    h1.fill(0.0);
    c1.fill(0.0);

    for (size_t t = 0; t < HISTORY_LEN; ++t) {
        std::vector<double> x0{history[t]};
        lstm_layer_forward(0, x0, h0, c0);
        std::vector<double> x1(h0.begin(), h0.end());
        lstm_layer_forward(1, x1, h1, c1);
    }
    return std::clamp(head_forward(h1), 0.0, 1.0);
}

void LSTMPredictor::online_update(const std::vector<double>& history, double target) {
    if (history.size() < HISTORY_LEN) return;

    std::array<double, HISTORY_LEN> arr{};
    for (size_t i = 0; i < HISTORY_LEN; ++i) arr[i] = history[i];
    const double pred = predict(arr);
    const double err = target - pred;

    // Adagrad on float-domain head weights + layer-0 input projection
    for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
        // Approximate gradient via last hidden ≈ err * head contribution
        const double g = err;
        ada_head_[i] += static_cast<float>(g * g);
        const double step = ADA_LR * g / (std::sqrt(ada_head_[i]) + ADA_EPS);
        double w = head_w_[i] * head_scale_ + step;
        // Re-quantize into INT8 + scale
        const float new_scale =
            std::max(head_scale_, static_cast<float>(std::abs(w) / 127.0 + 1e-8));
        head_scale_ = new_scale;
        head_w_[i] = static_cast<int8_t>(
            std::clamp(static_cast<int>(std::lround(w / head_scale_)), -127, 127));
    }
    head_b_ = static_cast<float>(head_b_ + ADA_LR * err);

    auto& w_ih = layers_[0].w_ih;
    for (size_t i = 0; i < w_ih.size() && i < ada_w_ih0_.size(); ++i) {
        const double g = err * 0.1;
        ada_w_ih0_[i] += static_cast<float>(g * g);
        const double step = ADA_LR * g / (std::sqrt(ada_w_ih0_[i]) + ADA_EPS);
        int q = static_cast<int>(w_ih[i]) + static_cast<int>(std::lround(step / layers_[0].scale_ih));
        w_ih[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
    }
}

void LSTMPredictor::update_cache(uint32_t edge_idx, double current_util,
                                 double predicted) {
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (edge_idx >= prediction_cache_.size()) {
        prediction_cache_.resize(edge_idx + 1, 0.0);
        current_util_cache_.resize(edge_idx + 1, 0.0);
    }
    current_util_cache_[edge_idx] = current_util;
    prediction_cache_[edge_idx] = predicted;
}

double LSTMPredictor::cached_prediction(uint32_t edge_idx) const {
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (edge_idx >= prediction_cache_.size()) return 0.0;
    return prediction_cache_[edge_idx];
}

double LSTMPredictor::effective_util(uint32_t edge_idx, double current_util,
                                     double transfer_duration_ms) const {
    const double predicted = cached_prediction(edge_idx);
    const double weight = std::min(transfer_duration_ms / PREDICT_AHEAD_MS, 1.0);
    const double blended = predicted * weight + current_util * (1.0 - weight);
    return std::max(current_util, blended);
}

void LSTMPredictor::push_sample(uint32_t edge_idx, double util) {
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (edge_idx >= edge_history_.size()) {
        edge_history_.resize(edge_idx + 1);
        edge_hist_count_.resize(edge_idx + 1, 0);
        for (size_t i = 0; i < edge_history_.size(); ++i) {
            if (edge_hist_count_[i] == 0) edge_history_[i].fill(0.0);
        }
    }
    auto& hist = edge_history_[edge_idx];
    const size_t n = edge_hist_count_[edge_idx];
    if (n < HISTORY_LEN) {
        hist[n] = util;
        edge_hist_count_[edge_idx] = n + 1;
    } else {
        for (size_t i = 1; i < HISTORY_LEN; ++i) hist[i - 1] = hist[i];
        hist[HISTORY_LEN - 1] = util;
    }
}

double LSTMPredictor::predict_edge(uint32_t edge_idx) const {
    std::array<double, HISTORY_LEN> hist{};
    hist.fill(0.0);
    {
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (edge_idx < edge_history_.size()) {
            hist = edge_history_[edge_idx];
        }
    }
    return predict(hist);
}

}  // namespace hics
