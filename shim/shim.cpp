#include "shim/shim.hpp"

#include <chrono>
#include <iostream>

namespace hics {

HICSShim::HICSShim() = default;
HICSShim::~HICSShim() {
    if (telemetry_) telemetry_->stop();
}

bool HICSShim::initialize(uint32_t num_nodes, uint32_t gpus_per_node) {
    graph_ = std::make_unique<TopologyGraph>(
        TopologyGraph::build_cluster(num_nodes, gpus_per_node));
    graph_->precompute_paths(6, 5);

    predictor_ = std::make_unique<LSTMPredictor>();
    telemetry_ = std::make_unique<TelemetryDaemon>(graph_->edges().size());
    path_engine_ = std::make_unique<PathSelectionEngine>(*graph_, *predictor_);

    run_profiling_sweep();
    telemetry_->start();
    return true;
}

void HICSShim::run_profiling_sweep() {
    // 200 ms profiling sweep to populate baseline bandwidth/latency
    using namespace std::chrono;
    auto start = steady_clock::now();
    while (duration_cast<milliseconds>(steady_clock::now() - start).count() < 200) {
        // Microbenchmark each edge — in production, measure B_ij and ℓ_ij
    }
}

PathCost HICSShim::dispatch_transfer(const TransferRequest& req) {
    auto t0 = std::chrono::steady_clock::now();

    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());

    // Update LSTM predictions per link
    for (size_t e = 0; e < graph_->edges().size(); ++e) {
        std::array<double, LSTMPredictor::HISTORY_LEN> history{};
        history.fill(static_cast<double>(utils[e]));
        double pred = predictor_->predict(history);
        predictor_->update_cache(static_cast<uint32_t>(e), utils[e], pred);
    }

    PathCost result;
    if (req.traffic_class == TrafficClass::KVMigration && req.size_bytes > 1024 * 1024 * 1024) {
        auto chunks = path_engine_->stripe_kv_migration(req, utils);
        if (!chunks.empty()) result = {chunks[0].edge_indices, 0.0};
        for (const auto& c : chunks) {
            result.latency_us += path_engine_->select_path(
                {req.source, req.dest, c.chunk_size, req.traffic_class, req.deadline_us},
                utils).latency_us;
        }
    } else {
        result = path_engine_->select_path(req, utils);
    }

    auto t1 = std::chrono::steady_clock::now();
    ++decisions_made_;
    total_decision_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    return result;
}

int HICSShim::intercept_collective(TrafficClass cls, EndpointId src, EndpointId dst,
                                     uint64_t size_bytes, double deadline_us) {
    TransferRequest req{src, dst, size_bytes, cls, deadline_us};
    auto path = dispatch_transfer(req);
    (void)path;
    return 0;
}

int HICSShim::intercept_p2p(TrafficClass cls, EndpointId src, EndpointId dst,
                             uint64_t size_bytes, double deadline_us) {
    return intercept_collective(cls, src, dst, size_bytes, deadline_us);
}

double HICSShim::avg_decision_latency_us() const {
    if (decisions_made_ == 0) return 0.0;
    return static_cast<double>(total_decision_ns_) / decisions_made_ / 1000.0;
}

}  // namespace hics

// NCCL plugin stubs
static hics::HICSShim* g_shim = nullptr;

extern "C" {

int hics_plugin_init() {
    g_shim = new hics::HICSShim();
    return g_shim->initialize() ? 0 : -1;
}

int hics_plugin_isend(void* send_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request) {
    (void)send_comm;
    (void)data;
    (void)tag;
    (void)mhandle;
    (void)request;
    if (!g_shim) return -1;
    return 0;
}

int hics_plugin_irecv(void* recv_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request) {
    (void)recv_comm;
    (void)data;
    (void)size;
    (void)tag;
    (void)mhandle;
    (void)request;
    return 0;
}

}  // extern "C"
