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
};

enum class TransferState : uint8_t {
    Pending = 0,
    InFlight = 1,
    Paused = 2,
    Rerouting = 3,
    Completed = 4,
    Cancelled = 5,
};

using TransferId = uint64_t;

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
    bool preemptible{true};  // KV migrations are preemptible; TP/PP are not
};

// Executes transfers along selected paths, tracks in-flight bytes, and
// supports SLO-driven preemption + reroute of KV migrations.
class TransferExecutor {
public:
    using CompletionFn = std::function<void(const TransferProgress&)>;

    TransferExecutor();

    void set_topology(const TopologyGraph* graph) { graph_ = graph; }
    void set_completion_callback(CompletionFn fn) { on_complete_ = std::move(fn); }

    // Submit a transfer on an already-selected path. Returns transfer id.
    TransferId submit(const TransferRequest& req, const PathCost& path);

    // Submit striped KV chunks as a single logical transfer group.
    TransferId submit_striped(const TransferRequest& req,
                              const std::vector<ChunkPath>& chunks);

    bool pause(TransferId id);
    bool resume(TransferId id, const PathCost& new_path);
    bool cancel(TransferId id);

    // Pause preemptible KV flows that share any of `busy_edges`, ordered by
    // slack descending. Returns ids that were paused.
    std::vector<TransferId> preempt_sharing(const std::vector<uint32_t>& busy_edges,
                                            size_t max_preempt = 8);

    // Reroute a paused transfer onto `new_path` and resume.
    bool reroute(TransferId id, const PathCost& new_path);

    // Advance simulated wire progress by `dt_ms` using path bottleneck BW
    // under `utils` contention. Returns newly completed transfer ids.
    std::vector<TransferId> tick(double dt_ms, const std::vector<float>& utils);

    // Immediate best-effort progress for decision-path demos (no wall clock).
    std::vector<TransferId> drain_ready(const std::vector<float>& utils);

    TransferProgress get(TransferId id) const;
    std::vector<TransferProgress> list_inflight() const;
    std::vector<TransferProgress> list_paused_kv() const;

    uint64_t completed_count() const { return completed_count_.load(); }
    uint64_t preempt_count() const { return preempt_count_.load(); }
    uint64_t reroute_count() const { return reroute_count_.load(); }
    uint64_t bytes_transferred() const { return bytes_transferred_.load(); }

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

    double bottleneck_gbps(const TransferProgress& tp,
                           const std::vector<float>& utils) const;
    uint64_t bytes_for_dt(const TransferProgress& tp, double dt_ms,
                          const std::vector<float>& utils) const;
    void complete_locked(TransferProgress& tp);
};

}  // namespace hics
