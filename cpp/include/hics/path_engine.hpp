#pragma once

#include "hics/lstm_predictor.hpp"
#include "hics/topology.hpp"
#include "hics/types.hpp"

#include <functional>
#include <queue>
#include <unordered_set>

namespace hics {

// Definition 1: Routing Flexibility Set F(t)
std::vector<std::vector<uint32_t>> flexibility_set(
    const TopologyGraph& graph, uint32_t src, uint32_t dst,
    TrafficClass traffic_class);

// Algorithm 1: HICS Path Selection
class PathSelectionEngine {
public:
    PathSelectionEngine(TopologyGraph& graph, LSTMPredictor& predictor);

    PathCost select_path(const TransferRequest& req,
                         const std::vector<float>& current_utils);

    // SLO-aware preemption (§V-F): pause KV migrations on shared links
    void preempt_kv_migrations(uint32_t src, uint32_t dst,
                               const std::vector<uint32_t>& path_edges);

    // Multi-path KV striping: 256 MB chunks (§VI)
    struct ChunkDispatch {
        uint32_t chunk_seq;
        std::vector<uint32_t> edge_indices;
        uint64_t chunk_size;
    };
    std::vector<ChunkDispatch> stripe_kv_migration(
        const TransferRequest& req, const std::vector<float>& current_utils,
        uint64_t chunk_size = 256 * 1024 * 1024);

    void set_utilization(const std::vector<float>& utils) { utils_ = utils; }

private:
    TopologyGraph& graph_;
    LSTMPredictor& predictor_;
    std::vector<float> utils_;

    double path_cost(const std::vector<uint32_t>& edge_indices,
                     uint64_t size_bytes, double transfer_duration_ms) const;

    struct KVMigrationFlow {
        uint32_t src;
        uint32_t dst;
        uint64_t bytes_remaining;
        double slack_us;
        bool paused;
    };
    std::vector<KVMigrationFlow> kv_flows_;
};

}  // namespace hics
