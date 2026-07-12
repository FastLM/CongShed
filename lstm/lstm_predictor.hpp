#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace hics {

// Binary weight file magic / version (matches Python export_int8_weights)
constexpr char LSTM_WEIGHT_MAGIC[8] = {'H', 'I', 'C', 'S', 'L', 'S', 'T', 'M'};
constexpr uint32_t LSTM_WEIGHT_VERSION = 2;

struct LstmLayerWeights {
    // PyTorch LSTM gate order: i, f, g, o — each of size H × input
    std::vector<int8_t> w_ih;   // [4H, input]
    std::vector<int8_t> w_hh;   // [4H, H]
    std::vector<float> b_ih;    // [4H]
    std::vector<float> b_hh;    // [4H]
    float scale_ih{0.01f};
    float scale_hh{0.01f};
    uint32_t input_size{1};
};

// 2-layer LSTM, 32 hidden units — INT8 inference runtime
// Predicts utilization 5 ms ahead from K=16 samples at 1 ms intervals
class LSTMPredictor {
public:
    static constexpr size_t HIDDEN_SIZE = 32;
    static constexpr size_t NUM_LAYERS = 2;
    static constexpr size_t HISTORY_LEN = 16;
    static constexpr size_t PREDICT_AHEAD_MS = 5;
    static constexpr size_t SAMPLE_INTERVAL_MS = 1;
    static constexpr size_t GATES = 4;  // i, f, g, o

    LSTMPredictor();

    // Load / save INT8-quantized weights (HICSLSTM v2 format)
    bool load_weights(const std::string& path);
    bool save_weights(const std::string& path) const;
    bool weights_loaded() const { return weights_loaded_; }

    // Online Adagrad update every ~50 ms (η=1e-3)
    void online_update(const std::vector<double>& history, double target);

    // Forward pass: predict utilization ∈ [0, 1]
    double predict(const std::array<double, HISTORY_LEN>& history) const;

    // Per-link prediction cache keyed by edge index
    void update_cache(uint32_t edge_idx, double current_util, double predicted);
    double cached_prediction(uint32_t edge_idx) const;
    double effective_util(uint32_t edge_idx, double current_util,
                          double transfer_duration_ms) const;

    // Per-edge rolling history for online prediction
    void push_sample(uint32_t edge_idx, double util);
    double predict_edge(uint32_t edge_idx) const;

    const std::vector<LstmLayerWeights>& layers() const { return layers_; }

private:
    std::vector<LstmLayerWeights> layers_;
    std::vector<int8_t> head_w_;
    float head_scale_{0.01f};
    float head_b_{0.0f};
    bool weights_loaded_{false};

    // Adagrad accumulators (float domain for head + layer-0 input weights)
    std::vector<float> ada_head_;
    std::vector<float> ada_w_ih0_;
    static constexpr double ADA_EPS = 1e-8;
    static constexpr double ADA_LR = 1e-3;

    mutable std::mutex cache_mu_;
    std::vector<double> prediction_cache_;
    std::vector<double> current_util_cache_;
    std::vector<std::array<double, HISTORY_LEN>> edge_history_;
    std::vector<size_t> edge_hist_count_;

    void init_default_weights();
    void lstm_layer_forward(size_t layer, const std::vector<double>& x,
                            std::array<double, HIDDEN_SIZE>& h,
                            std::array<double, HIDDEN_SIZE>& c) const;
    double head_forward(const std::array<double, HIDDEN_SIZE>& h) const;
};

}  // namespace hics
