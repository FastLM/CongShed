#include "path_engine/path_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hics {
namespace {

int path_rail_id(const TopologyGraph& graph, const std::vector<uint32_t>& path) {
    for (uint32_t ei : path) {
        if (ei >= graph.edges().size()) continue;
        const int r = graph.edges()[ei].attrs.rail_id;
        if (r >= 0) return r;
    }
    return -1;
}

double path_mean_util(const std::vector<uint32_t>& path,
                      const std::vector<float>& utils) {
    if (path.empty()) return 1.0;
    double sum = 0.0;
    size_t n = 0;
    for (uint32_t ei : path) {
        if (ei < utils.size()) {
            sum += utils[ei];
            ++n;
        }
    }
    return n ? sum / static_cast<double>(n) : 0.0;
}

}  // namespace

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
                    if (edge.attrs.fabric != FabricType::NVLink &&
                        edge.attrs.fabric != FabricType::InfiniBand) {
                        valid = false;
                    }
                    break;
                case TrafficClass::PipelineParallel:
                    break;
                case TrafficClass::KVMigration:
                    if (edge.attrs.fabric == FabricType::PCIe) {
                        valid = false;
                    }
                    break;
            }
            if (!valid) break;
        }
        if (valid) admissible.push_back(path);
    }

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
    if (candidates.empty()) {
        return {{}, std::numeric_limits<double>::max()};
    }

    if (candidates.size() == 1) {
        double dur_ms = path_cost(candidates[0], req.size_bytes, 1.0) / 1000.0;
        return {candidates[0], path_cost(candidates[0], req.size_bytes, dur_ms)};
    }

    double est_dur_ms = static_cast<double>(req.size_bytes) / 1e9 / 50.0 * 1000.0;

    PathCost best{{}, std::numeric_limits<double>::max()};
    for (const auto& path : candidates) {
        double cost = path_cost(path, req.size_bytes, est_dur_ms);
        // Telemetry-driven rail bias: prefer cooler IB rail for KV
        if (req.traffic_class == TrafficClass::KVMigration) {
            cost *= (0.75 + 0.5 * path_mean_util(path, current_utils));
        }
        if (cost < best.latency_us) {
            best = {path, cost};
        }
    }

    if (req.traffic_class == TrafficClass::TensorParallel &&
        best.latency_us > req.deadline_us) {
        preempt_kv_migrations(src, dst, best.edge_indices, current_utils, true);
    }

    return best;
}

void PathSelectionEngine::register_kv_flow(TransferId id, uint32_t src, uint32_t dst,
                                           uint64_t bytes, double slack_us,
                                           const std::vector<uint32_t>& edges) {
    kv_flows_.push_back({id, src, dst, bytes, slack_us, false, edges});
}

void PathSelectionEngine::unregister_kv_flow(TransferId id) {
    kv_flows_.erase(std::remove_if(kv_flows_.begin(), kv_flows_.end(),
                                   [id](const KVMigrationFlow& f) { return f.id == id; }),
                    kv_flows_.end());
}

std::vector<TransferId> PathSelectionEngine::preempt_kv_migrations(
    uint32_t src, uint32_t dst, const std::vector<uint32_t>& path_edges,
    const std::vector<float>& current_utils, bool do_reroute) {
    (void)src;
    (void)dst;
    std::vector<TransferId> paused_ids;

    if (executor_) {
        paused_ids = executor_->preempt_sharing(path_edges);
        preempt_events_ += paused_ids.size();

        if (do_reroute) {
            for (TransferId id : paused_ids) {
                auto prog = executor_->get(id);
                TransferRequest alt = prog.request;
                // Remaining bytes after completed chunks only
                uint64_t remain = prog.bytes_remaining;
                if (remain == 0) continue;

                auto utils = current_utils;
                for (uint32_t e : path_edges) {
                    if (e < utils.size()) utils[e] = std::min(0.99f, utils[e] + 0.5f);
                }
                uint32_t s = graph_.vertex_of(alt.source);
                uint32_t d = graph_.vertex_of(alt.dest);
                auto candidates = flexibility_set(graph_, s, d, alt.traffic_class);
                PathCost best{{}, std::numeric_limits<double>::max()};
                std::unordered_set<uint32_t> busy(path_edges.begin(), path_edges.end());
                const int busy_rail = path_rail_id(graph_, path_edges);

                for (const auto& path : candidates) {
                    bool overlaps = false;
                    for (uint32_t e : path) {
                        if (busy.count(e)) {
                            overlaps = true;
                            break;
                        }
                    }
                    double est = static_cast<double>(remain) / 1e9 / 50.0 * 1000.0;
                    utils_ = utils;
                    double cost = path_cost(path, remain, est);
                    if (!overlaps) cost *= 0.5;
                    // Prefer alternate rail when TP occupies the current one
                    const int rail = path_rail_id(graph_, path);
                    if (busy_rail >= 0 && rail >= 0 && rail != busy_rail) cost *= 0.6;
                    cost *= (0.75 + 0.5 * path_mean_util(path, utils));
                    if (cost < best.latency_us) best = {path, cost};
                }
                if (!best.edge_indices.empty() && executor_->reroute(id, best)) {
                    ++reroute_events_;
                    for (auto& flow : kv_flows_) {
                        if (flow.id == id) {
                            flow.paused = false;
                            flow.edges = best.edge_indices;
                            flow.bytes_remaining = remain;
                            flow.slack_us = best.latency_us > 0
                                                ? alt.deadline_us - best.latency_us
                                                : flow.slack_us;
                        }
                    }
                }
            }
        }
        return paused_ids;
    }

    std::unordered_set<uint32_t> shared(path_edges.begin(), path_edges.end());
    std::sort(kv_flows_.begin(), kv_flows_.end(),
              [](const KVMigrationFlow& a, const KVMigrationFlow& b) {
                  return a.slack_us > b.slack_us;
              });
    for (auto& flow : kv_flows_) {
        if (flow.paused || flow.bytes_remaining == 0) continue;
        bool overlap = false;
        for (uint32_t e : flow.edges) {
            if (shared.count(e)) {
                overlap = true;
                break;
            }
        }
        if (!overlap) continue;
        flow.paused = true;
        paused_ids.push_back(flow.id);
        ++preempt_events_;
    }
    return paused_ids;
}

std::vector<PathSelectionEngine::ChunkDispatch>
PathSelectionEngine::stripe_kv_migration(const TransferRequest& req,
                                          const std::vector<float>& current_utils,
                                          uint64_t chunk_size) {
    std::vector<ChunkDispatch> dispatches;
    if (chunk_size == 0) chunk_size = kKvChunkBytes;
    uint64_t remaining = req.size_bytes;
    uint32_t seq = 0;
    uint64_t offset = 0;

    uint32_t src = graph_.vertex_of(req.source);
    uint32_t dst = graph_.vertex_of(req.dest);
    auto candidates = flexibility_set(graph_, src, dst, req.traffic_class);
    if (candidates.empty()) return dispatches;

    utils_.assign(current_utils.begin(), current_utils.end());
    const double est_dur_ms =
        static_cast<double>(chunk_size) / 1e9 / 50.0 * 1000.0;

    // Per-chunk telemetry-driven path / rail selection (not blind round-robin)
    while (remaining > 0) {
        uint64_t this_chunk = std::min(remaining, chunk_size);
        PathCost best{{}, std::numeric_limits<double>::max()};
        for (const auto& path : candidates) {
            double cost = path_cost(path, this_chunk, est_dur_ms);
            cost *= (0.75 + 0.5 * path_mean_util(path, current_utils));
            // Spread consecutive chunks across rails when costs are close
            const int rail = path_rail_id(graph_, path);
            if (rail >= 0 && (seq % kIbRailCount) == static_cast<uint32_t>(rail)) {
                cost *= 0.92;
            }
            if (cost < best.latency_us) best = {path, cost};
        }
        if (best.edge_indices.empty()) best = {candidates[0], 0.0};

        ChunkDispatch d;
        d.chunk_seq = seq++;
        d.edge_indices = best.edge_indices;
        d.chunk_size = this_chunk;
        d.byte_offset = offset;
        d.rail_id = path_rail_id(graph_, best.edge_indices);
        dispatches.push_back(std::move(d));

        // Softly account for scheduled load so next chunk sees hotter rail
        for (uint32_t ei : best.edge_indices) {
            if (ei < utils_.size()) {
                utils_[ei] = std::min(0.99f, utils_[ei] + 0.05f);
            }
        }
        remaining -= this_chunk;
        offset += this_chunk;
    }
    return dispatches;
}

TransferId PathSelectionEngine::schedule_and_execute(
    const TransferRequest& req, const std::vector<float>& current_utils,
    const SubmitOptions& opts) {
    if (!executor_) return 0;

    TransferId id = 0;
    if (req.traffic_class == TrafficClass::KVMigration &&
        req.size_bytes > kKvChunkBytes) {
        auto chunks = stripe_kv_migration(req, current_utils, kKvChunkBytes);
        std::vector<ChunkPath> paths;
        paths.reserve(chunks.size());
        for (const auto& c : chunks) {
            paths.push_back(
                {c.chunk_seq, c.edge_indices, c.chunk_size, c.byte_offset, c.rail_id});
        }
        id = executor_->submit_striped(req, paths, opts);
        uint32_t src = graph_.vertex_of(req.source);
        uint32_t dst = graph_.vertex_of(req.dest);
        std::vector<uint32_t> edges;
        for (const auto& c : chunks)
            edges.insert(edges.end(), c.edge_indices.begin(), c.edge_indices.end());
        register_kv_flow(id, src, dst, req.size_bytes, req.deadline_us, edges);
    } else {
        PathCost path = select_path(req, current_utils);
        id = executor_->submit(req, path, opts);
        if (req.traffic_class == TrafficClass::KVMigration) {
            uint32_t src = graph_.vertex_of(req.source);
            uint32_t dst = graph_.vertex_of(req.dest);
            register_kv_flow(id, src, dst, req.size_bytes,
                             req.deadline_us - path.latency_us, path.edge_indices);
        }
    }
    return id;
}

}  // namespace hics
