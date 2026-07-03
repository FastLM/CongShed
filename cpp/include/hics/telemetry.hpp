#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace hics {

// Lock-free ring buffer shared via mmap, 4 KB single page
static constexpr size_t TELEMETRY_RING_SIZE = 512;
static constexpr size_t TELEMETRY_SAMPLE_HZ = 1000;

struct TelemetrySample {
    uint64_t timestamp_ns;
    uint32_t edge_idx;
    float utilization;  // ∈ [0, 1]
};

class TelemetryRingBuffer {
public:
    TelemetryRingBuffer();

    void write(uint32_t edge_idx, float utilization);
    bool read_latest(uint32_t edge_idx, float& out_util) const;
    std::vector<float> snapshot(size_t num_edges) const;

private:
    alignas(64) std::array<TelemetrySample, TELEMETRY_RING_SIZE> buffer_{};
    alignas(64) std::atomic<uint32_t> write_idx_{0};
};

// Hardware telemetry daemon: nvml, ibverbs MAD, PCIe PMU
class TelemetryDaemon {
public:
    explicit TelemetryDaemon(size_t num_edges);
    ~TelemetryDaemon();

    void start();
    void stop();

    TelemetryRingBuffer& ring_buffer() { return ring_; }
    const TelemetryRingBuffer& ring_buffer() const { return ring_; }

    // Inject utilization for simulation/testing
    void set_simulated_util(uint32_t edge_idx, float util);

private:
    size_t num_edges_;
    TelemetryRingBuffer ring_;
    std::thread collector_thread_;
    std::atomic<bool> running_{false};
    std::vector<float> simulated_utils_;

    void collect_loop();
    float read_nvlink_util(uint32_t edge_idx);
    float read_ib_util(uint32_t edge_idx);
    float read_pcie_util(uint32_t edge_idx);
};

}  // namespace hics
