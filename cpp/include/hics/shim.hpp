#pragma once

#include "hics/path_engine.hpp"
#include "hics/telemetry.hpp"

#include <chrono>
#include <functional>
#include <memory>

namespace hics {

// NCCL/UCX interposition shim (§V-A, §VI)
// Budget: <0.8 µs per decision
class HICSShim {
public:
    HICSShim();
    ~HICSShim();

    bool initialize(uint32_t num_nodes = 8, uint32_t gpus_per_node = 8);

    // Intercept collective (NCCL plugin hook)
    int intercept_collective(TrafficClass cls, EndpointId src, EndpointId dst,
                             uint64_t size_bytes, double deadline_us);

    // Intercept point-to-point (UCX transport hook)
    int intercept_p2p(TrafficClass cls, EndpointId src, EndpointId dst,
                      uint64_t size_bytes, double deadline_us);

    // Profiling sweep at startup (200 ms, §VI)
    void run_profiling_sweep();

    TelemetryDaemon& telemetry() { return *telemetry_; }
    PathSelectionEngine& path_engine() { return *path_engine_; }

    // Statistics
    uint64_t decisions_made() const { return decisions_made_; }
    double avg_decision_latency_us() const;

private:
    std::unique_ptr<TopologyGraph> graph_;
    std::unique_ptr<LSTMPredictor> predictor_;
    std::unique_ptr<TelemetryDaemon> telemetry_;
    std::unique_ptr<PathSelectionEngine> path_engine_;

    uint64_t decisions_made_{0};
    uint64_t total_decision_ns_{0};

    PathCost dispatch_transfer(const TransferRequest& req);
};

// NCCL plugin entry points (ncclNet_v8 API stubs)
extern "C" {
int hics_plugin_init();
int hics_plugin_isend(void* send_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request);
int hics_plugin_irecv(void* recv_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request);
}

}  // namespace hics
