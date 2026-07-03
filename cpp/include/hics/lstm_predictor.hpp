#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace hics {

// 2-layer LSTM, 32 hidden units (§V-C, Table III)
// Predicts utilization 5 ms ahead from K=16 samples at 1 ms intervals
class LSTMPredictor {
public:
    static constexpr size_t HIDDEN_SIZE = 32;
    static constexpr size_t HISTORY_LEN = 16;   // K samples
    static constexpr size_t PREDICT_AHEAD_MS = 5;
    static constexpr size_t SAMPLE_INTERVAL_MS = 1;

    LSTMPredictor();

    // Load INT8-quantized weights exported from PyTorch (§VI)
    bool load_weights(const std::string& path);

    // Online update with Adagrad η=1e-3 every 50 ms (§V-C)
    void online_update(const std::vector<double>& history, double target);

    // Forward pass: predict utilization ∈ [0, 1]
    double predict(const std::array<double, HISTORY_LEN>& history) const;

    // Per-link prediction cache keyed by edge index
    void update_cache(uint32_t edge_idx, double current_util, double predicted);
    double cached_prediction(uint32_t edge_idx) const;
    double effective_util(uint32_t edge_idx, double current_util,
                          double transfer_duration_ms) const;

private:
    // LSTM cell weights (simplified INT8 inference)
    std::vector<int8_t> w_ih_;  // input-to-hidden
    std::vector<int8_t> w_hh_;  // hidden-to-hidden
    std::vector<float> scale_ih_;
    std::vector<float> scale_hh_;
    mutable std::array<double, HIDDEN_SIZE> h_state_{};
    mutable std::array<double, HIDDEN_SIZE> c_state_{};

    std::vector<double> prediction_cache_;
    std::vector<double> current_util_cache_;

    void lstm_cell_forward(const std::array<double, 1>& input) const;
    void init_default_weights();
};

}  // namespace hics
