#include "hics/telemetry.hpp"

#include <chrono>
#include <cstring>
#include <thread>

namespace hics {

TelemetryRingBuffer::TelemetryRingBuffer() { buffer_.fill({0, 0, 0.0f}); }

void TelemetryRingBuffer::write(uint32_t edge_idx, float utilization) {
    uint32_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % TELEMETRY_RING_SIZE;
    buffer_[idx] = {static_cast<uint64_t>(
                        std::chrono::steady_clock::now().time_since_epoch().count()),
                    edge_idx, utilization};
    std::atomic_thread_fence(std::memory_order_release);
}

bool TelemetryRingBuffer::read_latest(uint32_t edge_idx, float& out_util) const {
    uint32_t start = write_idx_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < TELEMETRY_RING_SIZE; ++i) {
        uint32_t idx = (start + TELEMETRY_RING_SIZE - 1 - i) % TELEMETRY_RING_SIZE;
        if (buffer_[idx].edge_idx == edge_idx) {
            out_util = buffer_[idx].utilization;
            return true;
        }
    }
    return false;
}

std::vector<float> TelemetryRingBuffer::snapshot(size_t num_edges) const {
    std::vector<float> utils(num_edges, 0.0f);
    for (size_t e = 0; e < num_edges; ++e) {
        read_latest(static_cast<uint32_t>(e), utils[e]);
    }
    return utils;
}

TelemetryDaemon::TelemetryDaemon(size_t num_edges)
    : num_edges_(num_edges), simulated_utils_(num_edges, 0.0f) {}

TelemetryDaemon::~TelemetryDaemon() { stop(); }

void TelemetryDaemon::start() {
    if (running_.exchange(true)) return;
    collector_thread_ = std::thread([this] { collect_loop(); });
}

void TelemetryDaemon::stop() {
    if (!running_.exchange(false)) return;
    if (collector_thread_.joinable()) collector_thread_.join();
}

void TelemetryDaemon::set_simulated_util(uint32_t edge_idx, float util) {
    if (edge_idx < simulated_utils_.size()) {
        simulated_utils_[edge_idx] = std::clamp(util, 0.0f, 1.0f);
    }
}

void TelemetryDaemon::collect_loop() {
    using namespace std::chrono;
    const auto interval = microseconds(1000000 / TELEMETRY_SAMPLE_HZ);
    while (running_.load(std::memory_order_relaxed)) {
        for (size_t e = 0; e < num_edges_; ++e) {
            float util = simulated_utils_[e];
            // In production: read_nvlink_util / read_ib_util / read_pcie_util
            ring_.write(static_cast<uint32_t>(e), util);
        }
        std::this_thread::sleep_for(interval);
    }
}

float TelemetryDaemon::read_nvlink_util(uint32_t edge_idx) {
    // Production: nvmlDeviceGetNvLinkUtilizationCounter
    (void)edge_idx;
    return 0.0f;
}

float TelemetryDaemon::read_ib_util(uint32_t edge_idx) {
    // Production: ibverbs perfquery MAD batched every 2 ms
    (void)edge_idx;
    return 0.0f;
}

float TelemetryDaemon::read_pcie_util(uint32_t edge_idx) {
    // Production: pcm-iio PMU counters
    (void)edge_idx;
    return 0.0f;
}

}  // namespace hics
