#include "shim/shim.hpp"
#include "shim/ucx_transport.hpp"
#include "shim/ibverbs_transport.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace hics {

HICSShim::HICSShim() = default;

HICSShim::~HICSShim() {
    if (telemetry_) telemetry_->stop();
}

bool HICSShim::initialize(uint32_t num_nodes, uint32_t gpus_per_node) {
    InitOptions opts;
    opts.num_nodes = num_nodes;
    opts.gpus_per_node = gpus_per_node;
    return initialize(opts);
}

bool HICSShim::initialize(const InitOptions& opts) {
    graph_ = std::make_unique<TopologyGraph>(
        TopologyGraph::build_cluster(opts.num_nodes, opts.gpus_per_node));
    graph_->precompute_paths(6, 5);

    // Dual-rail ibverbs / host-memcpy backend for real KV buffer movement
    IbverbsTransport::instance().init(kIbRailCount);

    predictor_ = std::make_unique<LSTMPredictor>();
    if (!opts.weights_path.empty()) {
        if (!predictor_->load_weights(opts.weights_path)) {
            std::cerr << "HICS: failed to load LSTM weights from "
                      << opts.weights_path << " (using defaults)\n";
        }
    }

    std::vector<FabricType> fabrics;
    std::vector<double> peak_bw;
    std::vector<int> rail_ids;
    fabrics.reserve(graph_->edges().size());
    peak_bw.reserve(graph_->edges().size());
    rail_ids.reserve(graph_->edges().size());
    for (const auto& e : graph_->edges()) {
        fabrics.push_back(e.attrs.fabric);
        peak_bw.push_back(e.attrs.bandwidth_gbps);
        rail_ids.push_back(e.attrs.rail_id);
    }

    telemetry_ = std::make_unique<TelemetryDaemon>(graph_->edges().size(), opts.telemetry);
    telemetry_->configure_edges(fabrics, peak_bw, rail_ids);

    path_engine_ = std::make_unique<PathSelectionEngine>(*graph_, *predictor_);

    if (opts.enable_executor) {
        executor_ = std::make_unique<TransferExecutor>();
        executor_->set_topology(graph_.get());
        path_engine_->set_executor(executor_.get());
        executor_->set_completion_callback([this](const TransferProgress& tp) {
            if (tp.request.traffic_class == TrafficClass::KVMigration) {
                path_engine_->unregister_kv_flow(tp.id);
            }
        });
    }

    ucx_ = std::make_unique<UcxTransport>(*this);
    ucx_->open_iface({});

    run_profiling_sweep();
    if (opts.enable_telemetry) telemetry_->start();
    return true;
}

void HICSShim::run_profiling_sweep() {
    using namespace std::chrono;
    auto start = steady_clock::now();

    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());
    for (size_t ei = 0; ei < graph_->edges().size(); ++ei) {
        predictor_->push_sample(static_cast<uint32_t>(ei), utils[ei]);
    }
    // Warm the prediction cache for all edges once at startup
    for (size_t ei = 0; ei < graph_->edges().size(); ++ei) {
        double pred = predictor_->predict_edge(static_cast<uint32_t>(ei));
        predictor_->update_cache(static_cast<uint32_t>(ei), utils[ei], pred);
    }

    while (duration_cast<milliseconds>(steady_clock::now() - start).count() < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        refresh_predictions();
    }
}

void HICSShim::refresh_predictions() {
    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());
    for (size_t e = 0; e < graph_->edges().size(); ++e) {
        predictor_->push_sample(static_cast<uint32_t>(e), utils[e]);
    }
    // Full forward passes are amortized: predict a rotating window of edges
    // each call so the decision path stays off the critical LSTM cost.
    static thread_local size_t cursor = 0;
    constexpr size_t kBatch = 8;
    const size_t n = graph_->edges().size();
    if (n == 0) return;
    for (size_t i = 0; i < kBatch; ++i) {
        size_t e = (cursor + i) % n;
        double pred = predictor_->predict_edge(static_cast<uint32_t>(e));
        predictor_->update_cache(static_cast<uint32_t>(e), utils[e], pred);
    }
    cursor = (cursor + kBatch) % n;
}

PathCost HICSShim::dispatch_transfer(const TransferRequest& req) {
    auto t0 = std::chrono::steady_clock::now();
    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());

    PathCost result;
    if (req.traffic_class == TrafficClass::KVMigration &&
        req.size_bytes > kKvChunkBytes) {
        auto chunks = path_engine_->stripe_kv_migration(req, utils, kKvChunkBytes);
        if (!chunks.empty()) result = {chunks[0].edge_indices, 0.0};
        for (const auto& c : chunks) {
            result.latency_us += path_engine_
                                     ->select_path({req.source, req.dest, c.chunk_size,
                                                    req.traffic_class, req.deadline_us},
                                                   utils)
                                     .latency_us;
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

TransferId HICSShim::submit_transfer(TrafficClass cls, EndpointId src, EndpointId dst,
                                     uint64_t size_bytes, double deadline_us,
                                     void* src_buf, void* dst_buf) {
    TransferRequest req{src, dst, size_bytes, cls, deadline_us};
    auto t0 = std::chrono::steady_clock::now();
    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());

    TransferId id = 0;
    if (executor_ && path_engine_) {
        id = path_engine_->schedule_and_execute(req, utils, {src_buf, dst_buf});
        executor_->tick(0.05, utils);
    } else {
        dispatch_transfer(req);
    }

    auto t1 = std::chrono::steady_clock::now();
    ++decisions_made_;
    total_decision_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return id;
}

size_t HICSShim::poll_transfers(double dt_ms) {
    if (!executor_) return 0;
    auto utils = telemetry_->ring_buffer().snapshot(graph_->edges().size());
    auto done = executor_->tick(dt_ms, utils);
    return done.size();
}

TransferProgress HICSShim::transfer_status(TransferId id) const {
    if (!executor_) return {};
    return executor_->get(id);
}

int HICSShim::intercept_collective(TrafficClass cls, EndpointId src, EndpointId dst,
                                     uint64_t size_bytes, double deadline_us) {
    TransferId id = submit_transfer(cls, src, dst, size_bytes, deadline_us);
    return id == 0 ? -1 : 0;
}

int HICSShim::intercept_p2p(TrafficClass cls, EndpointId src, EndpointId dst,
                             uint64_t size_bytes, double deadline_us) {
    return intercept_collective(cls, src, dst, size_bytes, deadline_us);
}

double HICSShim::avg_decision_latency_us() const {
    if (decisions_made_ == 0) return 0.0;
    return static_cast<double>(total_decision_ns_) / decisions_made_ / 1000.0;
}

bool HICSShim::weights_loaded() const {
    return predictor_ && predictor_->weights_loaded();
}

}  // namespace hics
