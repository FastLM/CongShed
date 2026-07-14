#include "transfer/transfer_executor.hpp"
#include "shim/ibverbs_transport.hpp"

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

TransferExecutor::TransferExecutor() {
    IbverbsTransport::instance().init(kIbRailCount);
}

TransferId TransferExecutor::submit(const TransferRequest& req, const PathCost& path,
                                    const SubmitOptions& opts) {
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
    tp.src_buf = opts.src_buf;
    tp.dst_buf = opts.dst_buf;
    tp.chunk_quantum = kKvChunkBytes;

    ChunkProgress ch;
    ch.chunk_seq = 0;
    ch.edge_indices = path.edge_indices;
    ch.byte_offset = 0;
    ch.chunk_size = req.size_bytes;
    ch.bytes_sent = 0;
    ch.rail_id = rail_of_path(path.edge_indices);
    ch.state = TransferState::InFlight;
    tp.chunks.push_back(std::move(ch));
    tp.active_chunk = 0;

    transfers_[tp.id] = tp;
    return tp.id;
}

TransferId TransferExecutor::submit_striped(const TransferRequest& req,
                                            const std::vector<ChunkPath>& chunks,
                                            const SubmitOptions& opts) {
    if (chunks.empty()) {
        return submit(req, {}, opts);
    }

    std::lock_guard<std::mutex> lock(mu_);
    TransferProgress tp;
    tp.id = next_id_.fetch_add(1);
    tp.state = TransferState::InFlight;
    tp.request = req;
    tp.bytes_total = req.size_bytes;
    tp.bytes_sent = 0;
    tp.bytes_remaining = req.size_bytes;
    tp.started_ns = now_ns();
    tp.preemptible = (req.traffic_class == TrafficClass::KVMigration);
    tp.src_buf = opts.src_buf;
    tp.dst_buf = opts.dst_buf;
    tp.chunk_quantum = chunks.front().chunk_size > 0 ? chunks.front().chunk_size
                                                     : kKvChunkBytes;

    uint64_t offset = 0;
    // Start at most one chunk per IB rail; remainder stay Pending until a slot frees.
    std::vector<int> rail_inflight(kIbRailCount, 0);
    for (const auto& c : chunks) {
        ChunkProgress ch;
        ch.chunk_seq = c.chunk_seq;
        ch.edge_indices = c.edge_indices;
        ch.byte_offset = c.byte_offset > 0 ? c.byte_offset : offset;
        ch.chunk_size = c.chunk_size;
        ch.bytes_sent = 0;
        ch.rail_id = c.rail_id >= 0 ? c.rail_id : rail_of_path(c.edge_indices);
        const int slot = ch.rail_id >= 0 ? (ch.rail_id % kIbRailCount) : 0;
        if (rail_inflight[static_cast<size_t>(slot)] == 0) {
            ch.state = TransferState::InFlight;
            rail_inflight[static_cast<size_t>(slot)] = 1;
        } else {
            ch.state = TransferState::Pending;
        }
        for (uint32_t e : c.edge_indices) {
            if (std::find(tp.path_edges.begin(), tp.path_edges.end(), e) ==
                tp.path_edges.end()) {
                tp.path_edges.push_back(e);
            }
        }
        offset += c.chunk_size;
        tp.chunks.push_back(std::move(ch));
    }
    tp.slack_us = req.deadline_us;
    tp.active_chunk = 0;
    transfers_[tp.id] = tp;
    return tp.id;
}

int TransferExecutor::rail_of_path(const std::vector<uint32_t>& edges) const {
    if (!graph_) return 0;
    for (uint32_t ei : edges) {
        if (ei >= graph_->edges().size()) continue;
        const int r = graph_->edges()[ei].attrs.rail_id;
        if (r >= 0) return r;
    }
    return 0;
}

bool TransferExecutor::pause(TransferId id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(id);
    if (it == transfers_.end()) return false;
    auto& tp = it->second;
    if (!tp.preemptible) return false;
    if (tp.state != TransferState::InFlight &&
        tp.state != TransferState::SuspendPending) {
        return false;
    }
    // Chunk-boundary suspension: mark pending; tick() pauses after chunk completes
    tp.suspend_at_boundary = true;
    tp.state = TransferState::SuspendPending;
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
    for (auto& ch : it->second.chunks) {
        if (ch.state != TransferState::Completed) ch.state = TransferState::Cancelled;
    }
    return true;
}

std::vector<TransferId> TransferExecutor::preempt_sharing(
    const std::vector<uint32_t>& busy_edges, size_t max_preempt) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TransferProgress*> candidates;
    for (auto& kv : transfers_) {
        auto& tp = kv.second;
        if (!tp.preemptible) continue;
        if (tp.state != TransferState::InFlight) continue;
        bool hit = shares_edge(tp.path_edges, busy_edges);
        if (!hit) {
            for (const auto& ch : tp.chunks) {
                if (ch.state == TransferState::InFlight &&
                    shares_edge(ch.edge_indices, busy_edges)) {
                    hit = true;
                    break;
                }
            }
        }
        if (hit) candidates.push_back(&tp);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const TransferProgress* a, const TransferProgress* b) {
                  return a->slack_us > b->slack_us;
              });

    std::vector<TransferId> paused;
    for (size_t i = 0; i < candidates.size() && i < max_preempt; ++i) {
        candidates[i]->suspend_at_boundary = true;
        candidates[i]->state = TransferState::SuspendPending;
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
        tp.state != TransferState::InFlight &&
        tp.state != TransferState::SuspendPending) {
        return false;
    }
    tp.state = TransferState::Rerouting;
    const int new_rail = rail_of_path(new_path.edge_indices);
    // Only incomplete / future chunks get the new rail path
    for (auto& ch : tp.chunks) {
        if (ch.state == TransferState::Completed) continue;
        ch.edge_indices = new_path.edge_indices;
        ch.rail_id = new_rail;
        if (ch.state == TransferState::Paused || ch.state == TransferState::Pending ||
            ch.state == TransferState::InFlight) {
            ch.state = TransferState::Pending;
        }
    }
    // Activate one chunk per rail under the new path
    std::vector<int> rail_busy(kIbRailCount, 0);
    for (auto& ch : tp.chunks) {
        if (ch.state != TransferState::Pending) continue;
        const int slot = ch.rail_id >= 0 ? (ch.rail_id % kIbRailCount) : 0;
        if (rail_busy[static_cast<size_t>(slot)] == 0) {
            ch.state = TransferState::InFlight;
            rail_busy[static_cast<size_t>(slot)] = 1;
        }
    }
    tp.path_edges = new_path.edge_indices;
    tp.slack_us = tp.request.deadline_us - new_path.latency_us;
    tp.reroute_count += 1;
    tp.suspend_at_boundary = false;
    tp.state = TransferState::InFlight;
    reroute_count_.fetch_add(1);
    return true;
}

double TransferExecutor::bottleneck_gbps(const std::vector<uint32_t>& edges,
                                         const std::vector<float>& utils) const {
    if (!graph_ || edges.empty()) return 1.0;
    double best = 1e9;
    for (uint32_t ei : edges) {
        if (ei >= graph_->edges().size()) continue;
        const auto& attrs = graph_->edges()[ei].attrs;
        double u = ei < utils.size() ? utils[ei] : 0.0;
        u = std::clamp(u, 0.0, 0.99);
        const double residual = std::pow(1.0 - u, attrs.contention_alpha);
        best = std::min(best, attrs.bandwidth_gbps * residual);
    }
    return std::max(best, 0.1);
}

uint64_t TransferExecutor::bytes_for_dt(const std::vector<uint32_t>& edges,
                                        double dt_ms,
                                        const std::vector<float>& utils) const {
    const double gbps = bottleneck_gbps(edges, utils);
    const double bytes = gbps * 1e9 * (dt_ms / 1000.0);
    return static_cast<uint64_t>(std::max(0.0, bytes));
}

void TransferExecutor::maybe_wire_chunk(TransferProgress& tp, ChunkProgress& ch,
                                        uint64_t step) {
    if (step == 0 || !tp.src_buf || !tp.dst_buf) return;
    auto* src = static_cast<uint8_t*>(tp.src_buf) + ch.byte_offset + ch.bytes_sent;
    auto* dst = static_cast<uint8_t*>(tp.dst_buf) + ch.byte_offset + ch.bytes_sent;
    IbverbsTransport::instance().transfer_chunk(ch.rail_id, src, dst, step);
}

void TransferExecutor::refresh_aggregate_locked(TransferProgress& tp) {
    uint64_t sent = 0;
    for (const auto& ch : tp.chunks) sent += ch.bytes_sent;
    tp.bytes_sent = sent;
    tp.bytes_remaining = tp.bytes_total > sent ? tp.bytes_total - sent : 0;
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
        if (tp.state != TransferState::InFlight &&
            tp.state != TransferState::SuspendPending) {
            continue;
        }

        // On suspend: park Pending chunks; keep draining wire-active chunks
        // until each hits its 256 MB boundary.
        if (tp.suspend_at_boundary) {
            for (auto& ch : tp.chunks) {
                if (ch.state == TransferState::Pending) {
                    ch.state = TransferState::Paused;
                }
            }
        } else {
            // Promote Pending → InFlight when a rail slot is free (dual-rail)
            std::vector<int> rail_busy(kIbRailCount, 0);
            for (const auto& ch : tp.chunks) {
                if (ch.state == TransferState::InFlight) {
                    const int slot = ch.rail_id >= 0 ? (ch.rail_id % kIbRailCount) : 0;
                    rail_busy[static_cast<size_t>(slot)] = 1;
                }
            }
            for (auto& ch : tp.chunks) {
                if (ch.state != TransferState::Pending) continue;
                const int slot = ch.rail_id >= 0 ? (ch.rail_id % kIbRailCount) : 0;
                if (rail_busy[static_cast<size_t>(slot)] == 0) {
                    ch.state = TransferState::InFlight;
                    rail_busy[static_cast<size_t>(slot)] = 1;
                }
            }
        }

        // Advance in-flight chunks in parallel (one active chunk per rail)
        for (auto& ch : tp.chunks) {
            if (ch.state != TransferState::InFlight) continue;
            const uint64_t remain = ch.chunk_size - ch.bytes_sent;
            if (remain == 0) {
                ch.state = TransferState::Completed;
                continue;
            }
            uint64_t step = bytes_for_dt(ch.edge_indices, dt_ms, utils);
            if (step > remain) step = remain;
            maybe_wire_chunk(tp, ch, step);
            ch.bytes_sent += step;
            bytes_transferred_.fetch_add(step);
            if (ch.bytes_sent >= ch.chunk_size) {
                ch.bytes_sent = ch.chunk_size;
                ch.state = TransferState::Completed;
            }
        }

        refresh_aggregate_locked(tp);
        const double elapsed_us = (now_ns() - tp.started_ns) / 1000.0;
        tp.slack_us = tp.request.deadline_us - elapsed_us;

        if (tp.suspend_at_boundary) {
            bool draining = false;
            for (const auto& ch : tp.chunks) {
                if (ch.state == TransferState::InFlight) {
                    draining = true;
                    break;
                }
            }
            if (!draining) {
                tp.state = TransferState::Paused;
                tp.suspend_at_boundary = false;
                boundary_suspends_.fetch_add(1);
                continue;
            }
            tp.state = TransferState::SuspendPending;
        }

        if (tp.bytes_remaining == 0) {
            for (auto& ch : tp.chunks) ch.state = TransferState::Completed;
            complete_locked(tp);
            done.push_back(tp.id);
        }
    }
    IbverbsTransport::instance().poll(32);
    return done;
}

std::vector<TransferId> TransferExecutor::drain_ready(const std::vector<float>& utils) {
    std::vector<TransferId> all_done;
    for (int i = 0; i < 10000; ++i) {
        auto done = tick(1.0, utils);
        all_done.insert(all_done.end(), done.begin(), done.end());
        bool any = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto& kv : transfers_) {
                if (kv.second.state == TransferState::InFlight ||
                    kv.second.state == TransferState::SuspendPending) {
                    any = true;
                    break;
                }
            }
        }
        if (!any) break;
        if (done.empty()) {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& kv : transfers_) {
                auto& tp = kv.second;
                if (tp.state != TransferState::InFlight &&
                    tp.state != TransferState::SuspendPending) {
                    continue;
                }
                for (auto& ch : tp.chunks) {
                    if (ch.state != TransferState::InFlight) continue;
                    const uint64_t remain = ch.chunk_size - ch.bytes_sent;
                    maybe_wire_chunk(tp, ch, remain);
                    bytes_transferred_.fetch_add(remain);
                    ch.bytes_sent = ch.chunk_size;
                    ch.state = TransferState::Completed;
                }
                refresh_aggregate_locked(tp);
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
        if (kv.second.state == TransferState::InFlight ||
            kv.second.state == TransferState::SuspendPending) {
            out.push_back(kv.second);
        }
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
