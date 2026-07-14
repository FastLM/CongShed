#pragma once

#include "model/model.hpp"
#include "topology/topology.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hics {

struct ChunkPath {
    uint32_t chunk_seq{0};
    std::vector<uint32_t> edge_indices;
    uint64_t chunk_size{0};
    uint64_t byte_offset{0};
    int rail_id{-1};
};

enum class TransferState : uint8_t {
    Pending = 0,
    InFlight = 1,
    Paused = 2,
    Rerouting = 3,
    Completed = 4,
    Cancelled = 5,
    SuspendPending = 6,  // finish current chunk, then pause at boundary
};

using TransferId = uint64_t;

struct ChunkProgress {
    uint32_t chunk_seq{0};
    std::vector<uint32_t> edge_indices;
    uint64_t byte_offset{0};
    uint64_t chunk_size{0};
    uint64_t bytes_sent{0};
    int rail_id{-1};
    TransferState state{TransferState::Pending};
};

struct TransferProgress {
    TransferId id{0};
    TransferState state{TransferState::Pending};
    TransferRequest request{};
    std::vector<uint32_t> path_edges;
    uint64_t bytes_total{0};
    uint64_t bytes_sent{0};
    uint64_t bytes_remaining{0};
    double slack_us{0.0};
    double started_ns{0.0};
    uint32_t reroute_count{0};
    bool preemptible{true};
    // Real KV buffer endpoints (optional; host-visible for memcpy / RDMA)
    void* src_buf{nullptr};
    void* dst_buf{nullptr};
    uint64_t chunk_quantum{kKvChunkBytes};
    std::vector<ChunkProgress> chunks;
    uint32_t active_chunk{0};
    bool suspend_at_boundary{false};
};

struct SubmitOptions {
    void* src_buf{nullptr};
    void* dst_buf{nullptr};
};

// Executes transfers along selected paths, tracks in-flight bytes, and
// supports SLO-driven chunk-boundary preemption + rail reroute of KV migrations.
class TransferExecutor {
public:
    using CompletionFn = std::function<void(const TransferProgress&)>;

    TransferExecutor();

    void set_topology(const TopologyGraph* graph) { graph_ = graph; }
    void set_completion_callback(CompletionFn fn) { on_complete_ = std::move(fn); }

    TransferId submit(const TransferRequest& req, const PathCost& path,
                      const SubmitOptions& opts = {});

    // Submit striped KV chunks as parallel per-chunk flows under one parent id.
    TransferId submit_striped(const TransferRequest& req,
                              const std::vector<ChunkPath>& chunks,
                              const SubmitOptions& opts = {});

    bool pause(TransferId id);
    bool resume(TransferId id, const PathCost& new_path);
    bool cancel(TransferId id);

    // Request suspend at next chunk boundary for preemptible flows sharing edges.
    std::vector<TransferId> preempt_sharing(const std::vector<uint32_t>& busy_edges,
                                            size_t max_preempt = 8);

    // Reroute remaining (incomplete) chunks onto `new_path` and resume.
    bool reroute(TransferId id, const PathCost& new_path);

    // Advance progress by `dt_ms`; moves real KV bytes when buffers are bound.
    std::vector<TransferId> tick(double dt_ms, const std::vector<float>& utils);

    std::vector<TransferId> drain_ready(const std::vector<float>& utils);

    TransferProgress get(TransferId id) const;
    std::vector<TransferProgress> list_inflight() const;
    std::vector<TransferProgress> list_paused_kv() const;

    uint64_t completed_count() const { return completed_count_.load(); }
    uint64_t preempt_count() const { return preempt_count_.load(); }
    uint64_t reroute_count() const { return reroute_count_.load(); }
    uint64_t bytes_transferred() const { return bytes_transferred_.load(); }
    uint64_t boundary_suspends() const { return boundary_suspends_.load(); }

private:
    const TopologyGraph* graph_{nullptr};
    CompletionFn on_complete_;
    mutable std::mutex mu_;
    std::unordered_map<TransferId, TransferProgress> transfers_;
    std::atomic<uint64_t> next_id_{1};
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<uint64_t> preempt_count_{0};
    std::atomic<uint64_t> reroute_count_{0};
    std::atomic<uint64_t> bytes_transferred_{0};
    std::atomic<uint64_t> boundary_suspends_{0};

    double bottleneck_gbps(const std::vector<uint32_t>& edges,
                           const std::vector<float>& utils) const;
    uint64_t bytes_for_dt(const std::vector<uint32_t>& edges, double dt_ms,
                          const std::vector<float>& utils) const;
    int rail_of_path(const std::vector<uint32_t>& edges) const;
    void refresh_aggregate_locked(TransferProgress& tp);
    void complete_locked(TransferProgress& tp);
    void maybe_wire_chunk(TransferProgress& tp, ChunkProgress& ch, uint64_t step);
};

}  // namespace hics
