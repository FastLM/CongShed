#include "transfer/transfer_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace hics {
namespace {

double now_ns() {
    return static_cast<double>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool shares_edge(const std::vector<uint32_t>& path,
                 const std::vector<uint32_t>& busy) {
    for (uint32_t e : path) {
        if (std::find(busy.begin(), busy.end(), e) != busy.end()) return true;
    }
    return false;
}

}  // namespace

TransferExecutor::TransferExecutor() = default;

TransferId TransferExecutor::submit(const TransferRequest& req, const PathCost& path) {
    std::lock_guard<std::mutex> lock(mu_);
    TransferProgress tp;
    tp.id = next_id_.fetch_add(1);
    tp.state = TransferState::InFlight;
    tp.request = req;
    tp.path_edges = path.edge_indices;
    tp.bytes_total = req.size_bytes;
    tp.bytes_sent = 0;
    tp.bytes_remaining = req.size_bytes;
    tp.slack_us = req.deadline_us - path.latency_us;
    tp.started_ns = now_ns();
    tp.preemptible = (req.traffic_class == TrafficClass::KVMigration);
    transfers_[tp.id] = tp;
    return tp.id;
}

TransferId TransferExecutor::submit_striped(
    const TransferRequest& req,
    const std::vector<ChunkPath>& chunks) {
    // Represent striped transfer as one logical flow using the first chunk path;
    // tick() still moves total bytes. Chunk paths are merged for conflict checks.
    PathCost merged;
    merged.latency_us = 0.0;
    for (const auto& c : chunks) {
        merged.edge_indices.insert(merged.edge_indices.end(),
                                   c.edge_indices.begin(), c.edge_indices.end());
        // De-duplicate while preserving order
    }
    std::vector<uint32_t> uniq;
    for (uint32_t e : merged.edge_indices) {
        if (std::find(uniq.begin(), uniq.end(), e) == uniq.end()) uniq.push_back(e);
    }
    merged.edge_indices = std::move(uniq);
    return submit(req, merged);
}

bool TransferExecutor::pause(TransferId id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(id);
    if (it == transfers_.end()) return false;
    if (it->second.state != TransferState::InFlight) return false;
    if (!it->second.preemptible) return false;
    it->second.state = TransferState::Paused;
    preempt_count_.fetch_add(1);
    return true;
}

bool TransferExecutor::resume(TransferId id, const PathCost& new_path) {
    return reroute(id, new_path);
}

bool TransferExecutor::cancel(TransferId id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(id);
    if (it == transfers_.end()) return false;
    it->second.state = TransferState::Cancelled;
    it->second.bytes_remaining = 0;
    return true;
}

std::vector<TransferId> TransferExecutor::preempt_sharing(
    const std::vector<uint32_t>& busy_edges, size_t max_preempt) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TransferProgress*> candidates;
    for (auto& kv : transfers_) {
        auto& tp = kv.second;
        if (tp.state != TransferState::InFlight || !tp.preemptible) continue;
        if (shares_edge(tp.path_edges, busy_edges)) candidates.push_back(&tp);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const TransferProgress* a, const TransferProgress* b) {
                  return a->slack_us > b->slack_us;
              });

    std::vector<TransferId> paused;
    for (size_t i = 0; i < candidates.size() && i < max_preempt; ++i) {
        candidates[i]->state = TransferState::Paused;
        paused.push_back(candidates[i]->id);
        preempt_count_.fetch_add(1);
    }
    return paused;
}

bool TransferExecutor::reroute(TransferId id, const PathCost& new_path) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(id);
    if (it == transfers_.end()) return false;
    auto& tp = it->second;
    if (tp.state != TransferState::Paused && tp.state != TransferState::Rerouting &&
        tp.state != TransferState::InFlight) {
        return false;
    }
    tp.state = TransferState::Rerouting;
    tp.path_edges = new_path.edge_indices;
    tp.slack_us = tp.request.deadline_us - new_path.latency_us;
    tp.reroute_count += 1;
    tp.state = TransferState::InFlight;
    reroute_count_.fetch_add(1);
    return true;
}

double TransferExecutor::bottleneck_gbps(const TransferProgress& tp,
                                         const std::vector<float>& utils) const {
    if (!graph_ || tp.path_edges.empty()) return 1.0;
    double best = 1e9;
    for (uint32_t ei : tp.path_edges) {
        if (ei >= graph_->edges().size()) continue;
        const auto& attrs = graph_->edges()[ei].attrs;
        double u = ei < utils.size() ? utils[ei] : 0.0;
        u = std::clamp(u, 0.0, 0.99);
        const double residual = std::pow(1.0 - u, attrs.contention_alpha);
        best = std::min(best, attrs.bandwidth_gbps * residual);
    }
    return std::max(best, 0.1);
}

uint64_t TransferExecutor::bytes_for_dt(const TransferProgress& tp, double dt_ms,
                                        const std::vector<float>& utils) const {
    const double gbps = bottleneck_gbps(tp, utils);
    const double bytes = gbps * 1e9 * (dt_ms / 1000.0);
    return static_cast<uint64_t>(std::max(0.0, bytes));
}

void TransferExecutor::complete_locked(TransferProgress& tp) {
    tp.state = TransferState::Completed;
    tp.bytes_remaining = 0;
    completed_count_.fetch_add(1);
    if (on_complete_) on_complete_(tp);
}

std::vector<TransferId> TransferExecutor::tick(double dt_ms,
                                               const std::vector<float>& utils) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TransferId> done;
    for (auto& kv : transfers_) {
        auto& tp = kv.second;
        if (tp.state != TransferState::InFlight) continue;
        uint64_t step = bytes_for_dt(tp, dt_ms, utils);
        if (step > tp.bytes_remaining) step = tp.bytes_remaining;
        tp.bytes_sent += step;
        tp.bytes_remaining -= step;
        bytes_transferred_.fetch_add(step);
        // Update slack from elapsed vs deadline
        const double elapsed_us = (now_ns() - tp.started_ns) / 1000.0;
        tp.slack_us = tp.request.deadline_us - elapsed_us;
        if (tp.bytes_remaining == 0) {
            complete_locked(tp);
            done.push_back(tp.id);
        }
    }
    return done;
}

std::vector<TransferId> TransferExecutor::drain_ready(const std::vector<float>& utils) {
    // Advance in 1 ms quanta until no in-flight work remains or a safety cap hits
    std::vector<TransferId> all_done;
    for (int i = 0; i < 10000; ++i) {
        auto done = tick(1.0, utils);
        all_done.insert(all_done.end(), done.begin(), done.end());
        bool any = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto& kv : transfers_) {
                if (kv.second.state == TransferState::InFlight) {
                    any = true;
                    break;
                }
            }
        }
        if (!any) break;
        // For tiny messages, finish in one step
        if (done.empty()) {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& kv : transfers_) {
                auto& tp = kv.second;
                if (tp.state != TransferState::InFlight) continue;
                bytes_transferred_.fetch_add(tp.bytes_remaining);
                tp.bytes_sent = tp.bytes_total;
                complete_locked(tp);
                all_done.push_back(tp.id);
            }
            break;
        }
    }
    return all_done;
}

TransferProgress TransferExecutor::get(TransferId id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(id);
    if (it == transfers_.end()) return {};
    return it->second;
}

std::vector<TransferProgress> TransferExecutor::list_inflight() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TransferProgress> out;
    for (const auto& kv : transfers_) {
        if (kv.second.state == TransferState::InFlight) out.push_back(kv.second);
    }
    return out;
}

std::vector<TransferProgress> TransferExecutor::list_paused_kv() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TransferProgress> out;
    for (const auto& kv : transfers_) {
        if (kv.second.state == TransferState::Paused && kv.second.preemptible)
            out.push_back(kv.second);
    }
    return out;
}

}  // namespace hics
