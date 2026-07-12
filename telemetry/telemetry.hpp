#pragma once

#include "model/model.hpp"
#include "telemetry/hw_counters.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hics {

static constexpr size_t TELEMETRY_RING_SIZE = 512;
static constexpr size_t TELEMETRY_SAMPLE_HZ = 1000;

struct TelemetrySample {
    uint64_t timestamp_ns;
    uint32_t edge_idx;
    float utilization;
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

struct TelemetryConfig {
    TelemetryBackendKind preferred{TelemetryBackendKind::Auto};
    bool allow_synthetic_fallback{true};
    uint32_t sample_hz{TELEMETRY_SAMPLE_HZ};
};

// Hardware telemetry daemon: NVML NVLink, IB sysfs MAD counters, PCIe sysfs,
// with synthetic fallback when counters are unavailable.
class TelemetryDaemon {
public:
    TelemetryDaemon(size_t num_edges, TelemetryConfig config = {});
    ~TelemetryDaemon();

    void start();
    void stop();

    // Bind edges to fabric types / peak BW before start()
    void configure_edges(const std::vector<FabricType>& fabrics,
                         const std::vector<double>& peak_bw_gbps);

    TelemetryRingBuffer& ring_buffer() { return ring_; }
    const TelemetryRingBuffer& ring_buffer() const { return ring_; }

    // Inject utilization for tests (overrides next hardware sample for that edge)
    void set_simulated_util(uint32_t edge_idx, float util);

    const char* active_backend_name() const;
    bool using_hardware() const { return using_hardware_; }

private:
    size_t num_edges_;
    TelemetryConfig config_;
    TelemetryRingBuffer ring_;
    std::thread collector_thread_;
    std::atomic<bool> running_{false};

    std::vector<float> override_utils_;
    std::vector<uint8_t> override_valid_;
    std::vector<EdgeHwBinding> bindings_;
    std::vector<std::unique_ptr<HwCounterBackend>> backends_;
    std::unique_ptr<HwCounterBackend> synthetic_;
    bool using_hardware_{false};
    std::string active_name_{"none"};
    mutable std::mutex mu_;

    void collect_loop();
    void select_backends();
    float read_edge_util(uint32_t edge_idx);
};

}  // namespace hics
