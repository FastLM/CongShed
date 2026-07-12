#pragma once

#include "lstm/lstm_predictor.hpp"
#include "topology/topology.hpp"
#include "model/model.hpp"
#include "transfer/transfer_executor.hpp"

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace hics {

std::vector<std::vector<uint32_t>> flexibility_set(
    const TopologyGraph& graph, uint32_t src, uint32_t dst,
    TrafficClass traffic_class);

class PathSelectionEngine {
public:
    PathSelectionEngine(TopologyGraph& graph, LSTMPredictor& predictor);

    void set_executor(TransferExecutor* exec) { executor_ = exec; }
    TransferExecutor* executor() const { return executor_; }

    PathCost select_path(const TransferRequest& req,
                         const std::vector<float>& current_utils);

    // Register / unregister KV flows for SLO preemption bookkeeping
    void register_kv_flow(TransferId id, uint32_t src, uint32_t dst,
                          uint64_t bytes, double slack_us,
                          const std::vector<uint32_t>& edges);
    void unregister_kv_flow(TransferId id);

    // Pause KV migrations sharing path edges; optionally reroute them
    std::vector<TransferId> preempt_kv_migrations(
        uint32_t src, uint32_t dst, const std::vector<uint32_t>& path_edges,
        const std::vector<float>& current_utils, bool reroute = true);

    struct ChunkDispatch {
        uint32_t chunk_seq;
        std::vector<uint32_t> edge_indices;
        uint64_t chunk_size;
    };
    std::vector<ChunkDispatch> stripe_kv_migration(
        const TransferRequest& req, const std::vector<float>& current_utils,
        uint64_t chunk_size = 256 * 1024 * 1024);

    // Full schedule: select path(s), submit to executor, preempt if needed
    TransferId schedule_and_execute(const TransferRequest& req,
                                    const std::vector<float>& current_utils);

    void set_utilization(const std::vector<float>& utils) { utils_ = utils; }

    uint64_t preempt_events() const { return preempt_events_; }
    uint64_t reroute_events() const { return reroute_events_; }

private:
    TopologyGraph& graph_;
    LSTMPredictor& predictor_;
    TransferExecutor* executor_{nullptr};
    std::vector<float> utils_;
    uint64_t preempt_events_{0};
    uint64_t reroute_events_{0};

    double path_cost(const std::vector<uint32_t>& edge_indices,
                     uint64_t size_bytes, double transfer_duration_ms) const;

    struct KVMigrationFlow {
        TransferId id{0};
        uint32_t src{0};
        uint32_t dst{0};
        uint64_t bytes_remaining{0};
        double slack_us{0.0};
        bool paused{false};
        std::vector<uint32_t> edges;
    };
    std::vector<KVMigrationFlow> kv_flows_;
};

}  // namespace hics
