#include "path_engine/path_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hics {

std::vector<std::vector<uint32_t>> flexibility_set(const TopologyGraph& graph,
                                                    uint32_t src, uint32_t dst,
                                                    TrafficClass traffic_class) {
    const auto& all_paths = graph.paths(src, dst);
    std::vector<std::vector<uint32_t>> admissible;

    for (const auto& path : all_paths) {
        bool valid = true;
        for (uint32_t ei : path) {
            const auto& edge = graph.edges()[ei];
            switch (traffic_class) {
                case TrafficClass::TensorParallel:
                    // TP must use NVLink intra-node or IB inter-node
                    if (edge.attrs.fabric != FabricType::NVLink &&
                        edge.attrs.fabric != FabricType::InfiniBand) {
                        valid = false;
                    }
                    break;
                case TrafficClass::PipelineParallel:
                    // PP can use any fabric
                    break;
                case TrafficClass::KVMigration:
                    // KV most flexible: NVLink, IB, or CXL
                    if (edge.attrs.fabric == FabricType::PCIe) {
                        // Avoid raw PCIe unless no alternative
                        valid = false;
                    }
                    break;
            }
            if (!valid) break;
        }
        if (valid) admissible.push_back(path);
    }

    // Fallback: allow PCIe for KV if no other path
    if (admissible.empty() && traffic_class == TrafficClass::KVMigration) {
        return std::vector<std::vector<uint32_t>>(all_paths.begin(), all_paths.end());
    }
    return admissible;
}

PathSelectionEngine::PathSelectionEngine(TopologyGraph& graph, LSTMPredictor& predictor)
    : graph_(graph), predictor_(predictor) {}

double PathSelectionEngine::path_cost(const std::vector<uint32_t>& edge_indices,
                                       uint64_t size_bytes,
                                       double transfer_duration_ms) const {
    double total = 0.0;
    for (uint32_t ei : edge_indices) {
        const auto& edge = graph_.edges()[ei];
        double util = ei < utils_.size() ? utils_[ei] : 0.0;
        util = predictor_.effective_util(ei, util, transfer_duration_ms);
        total += edge_latency_us(edge.attrs, size_bytes, util);
    }
    return total;
}

PathCost PathSelectionEngine::select_path(const TransferRequest& req,
                                           const std::vector<float>& current_utils) {
    utils_.assign(current_utils.begin(), current_utils.end());

    uint32_t src = graph_.vertex_of(req.source);
    uint32_t dst = graph_.vertex_of(req.dest);
    if (src == UINT32_MAX || dst == UINT32_MAX) {
        return {{}, std::numeric_limits<double>::max()};
    }

    auto candidates = flexibility_set(graph_, src, dst, req.traffic_class);

    // Single path → no overhead
    if (candidates.size() <= 1) {
        if (candidates.empty()) return {{}, std::numeric_limits<double>::max()};
        double dur_ms = path_cost(candidates[0], req.size_bytes, 1.0) / 1000.0;
        return {candidates[0], path_cost(candidates[0], req.size_bytes, dur_ms)};
    }

    // Estimate transfer duration for forward-looking util
    double est_dur_ms = static_cast<double>(req.size_bytes) / 1e9 / 50.0 * 1000.0;

    PathCost best{{}, std::numeric_limits<double>::max()};
    for (const auto& path : candidates) {
        double cost = path_cost(path, req.size_bytes, est_dur_ms);
        if (cost < best.latency_us) {
            best = {path, cost};
        }
    }

    // TP preemption when deadline exceeded
    if (req.traffic_class == TrafficClass::TensorParallel &&
        best.latency_us > req.deadline_us) {
        preempt_kv_migrations(src, dst, best.edge_indices);
    }

    return best;
}

void PathSelectionEngine::preempt_kv_migrations(
    uint32_t src, uint32_t dst, const std::vector<uint32_t>& path_edges) {
    std::unordered_set<uint32_t> shared(path_edges.begin(), path_edges.end());

    // Sort by slack descending — preempt largest slack first
    std::sort(kv_flows_.begin(), kv_flows_.end(),
              [](const KVMigrationFlow& a, const KVMigrationFlow& b) {
                  return a.slack_us > b.slack_us;
              });

    for (auto& flow : kv_flows_) {
        if (flow.paused || flow.bytes_remaining == 0) continue;
        // Pause flows that share congested links
        flow.paused = true;
        (void)src;
        (void)dst;
        (void)shared;
    }
}

std::vector<PathSelectionEngine::ChunkDispatch>
PathSelectionEngine::stripe_kv_migration(const TransferRequest& req,
                                          const std::vector<float>& current_utils,
                                          uint64_t chunk_size) {
    std::vector<ChunkDispatch> dispatches;
    uint64_t remaining = req.size_bytes;
    uint32_t seq = 0;

    while (remaining > 0) {
        uint64_t this_chunk = std::min(remaining, chunk_size);
        TransferRequest chunk_req = req;
        chunk_req.size_bytes = this_chunk;

        PathCost path = select_path(chunk_req, current_utils);
        dispatches.push_back({seq++, path.edge_indices, this_chunk});
        remaining -= this_chunk;
    }
    return dispatches;
}

}  // namespace hics
