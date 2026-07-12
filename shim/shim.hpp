#pragma once

#include "path_engine/path_engine.hpp"
#include "telemetry/telemetry.hpp"
#include "transfer/transfer_executor.hpp"

#include <chrono>
#include <memory>
#include <string>

namespace hics {

class UcxTransport;

// NCCL/UCX interposition shim — decision budget < 0.8 µs
class HICSShim {
public:
    struct InitOptions {
        uint32_t num_nodes{8};
        uint32_t gpus_per_node{8};
        std::string weights_path;  // INT8 LSTM weights (optional)
        bool enable_executor{true};
        bool enable_telemetry{true};
        TelemetryConfig telemetry{};
    };

    HICSShim();
    ~HICSShim();

    bool initialize(uint32_t num_nodes = 8, uint32_t gpus_per_node = 8);
    bool initialize(const InitOptions& opts);

    // Intercept collective (NCCL plugin hook)
    int intercept_collective(TrafficClass cls, EndpointId src, EndpointId dst,
                             uint64_t size_bytes, double deadline_us);

    // Intercept point-to-point (UCX transport hook)
    int intercept_p2p(TrafficClass cls, EndpointId src, EndpointId dst,
                      uint64_t size_bytes, double deadline_us);

    // Schedule + execute a transfer; returns transfer id
    TransferId submit_transfer(TrafficClass cls, EndpointId src, EndpointId dst,
                               uint64_t size_bytes, double deadline_us);

    // Progress in-flight transfers by dt_ms; returns completions
    size_t poll_transfers(double dt_ms = 0.2);
    TransferProgress transfer_status(TransferId id) const;

    // Profiling sweep at startup (~200 ms)
    void run_profiling_sweep();

    TelemetryDaemon& telemetry() { return *telemetry_; }
    PathSelectionEngine& path_engine() { return *path_engine_; }
    TransferExecutor& executor() { return *executor_; }
    LSTMPredictor& predictor() { return *predictor_; }
    TopologyGraph& graph() { return *graph_; }
    UcxTransport& ucx() { return *ucx_; }

    uint64_t decisions_made() const { return decisions_made_; }
    double avg_decision_latency_us() const;
    bool weights_loaded() const;

private:
    std::unique_ptr<TopologyGraph> graph_;
    std::unique_ptr<LSTMPredictor> predictor_;
    std::unique_ptr<TelemetryDaemon> telemetry_;
    std::unique_ptr<PathSelectionEngine> path_engine_;
    std::unique_ptr<TransferExecutor> executor_;
    std::unique_ptr<UcxTransport> ucx_;

    uint64_t decisions_made_{0};
    uint64_t total_decision_ns_{0};

    PathCost dispatch_transfer(const TransferRequest& req);
    void refresh_predictions();
};

}  // namespace hics
