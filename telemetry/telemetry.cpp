#include "telemetry/telemetry.hpp"
#include "telemetry/ib_mad.hpp"
#include "telemetry/pmu_counters.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace hics {

TelemetryRingBuffer::TelemetryRingBuffer() { buffer_.fill({0, 0, 0.0f}); }

void TelemetryRingBuffer::write(uint32_t edge_idx, float utilization) {
    uint32_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % TELEMETRY_RING_SIZE;
    buffer_[idx] = {
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()),
        edge_idx, utilization};
    std::atomic_thread_fence(std::memory_order_release);
}

bool TelemetryRingBuffer::read_latest(uint32_t edge_idx, float& out_util) const {
    uint32_t start = write_idx_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < TELEMETRY_RING_SIZE; ++i) {
        uint32_t idx = (start + TELEMETRY_RING_SIZE - 1 - i) % TELEMETRY_RING_SIZE;
        if (buffer_[idx].edge_idx == edge_idx && buffer_[idx].timestamp_ns != 0) {
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

TelemetryDaemon::TelemetryDaemon(size_t num_edges, TelemetryConfig config)
    : num_edges_(num_edges),
      config_(config),
      override_utils_(num_edges, 0.0f),
      override_valid_(num_edges, 0) {
    synthetic_ = make_synthetic_backend();
}

TelemetryDaemon::~TelemetryDaemon() { stop(); }

void TelemetryDaemon::configure_edges(const std::vector<FabricType>& fabrics,
                                      const std::vector<double>& peak_bw_gbps) {
    std::lock_guard<std::mutex> lock(mu_);
    bindings_ = bindings_from_fabrics(fabrics, peak_bw_gbps);
    if (bindings_.size() < num_edges_) {
        // Pad with IB defaults
        std::vector<FabricType> f = fabrics;
        std::vector<double> bw = peak_bw_gbps;
        while (f.size() < num_edges_) {
            f.push_back(FabricType::InfiniBand);
            bw.push_back(50.0);
        }
        bindings_ = bindings_from_fabrics(f, bw);
    }
    select_backends();
}

void TelemetryDaemon::select_backends() {
    backends_.clear();
    using_hardware_ = false;

    auto try_add = [&](std::unique_ptr<HwCounterBackend> b) {
        if (!b || !b->available()) return;
        if (!b->init(bindings_)) return;
        active_name_ = b->name();
        if (std::strcmp(b->name(), "synthetic") != 0) using_hardware_ = true;
        backends_.push_back(std::move(b));
    };

    switch (config_.preferred) {
        case TelemetryBackendKind::Nvml:
            try_add(make_nvml_backend());
            break;
        case TelemetryBackendKind::Ibverbs:
            try_add(make_ibverbs_backend());
            break;
        case TelemetryBackendKind::IbMad:
            try_add(make_ib_mad_backend());
            break;
        case TelemetryBackendKind::PcieSysfs:
            try_add(make_pcie_sysfs_backend());
            break;
        case TelemetryBackendKind::Pmu:
            try_add(make_pmu_backend());
            break;
        case TelemetryBackendKind::Synthetic:
            break;
        case TelemetryBackendKind::Auto:
        default:
            try_add(make_nvml_backend());
            try_add(make_ib_mad_backend());   // MAD first, sysfs fallback inside
            try_add(make_ibverbs_backend());
            try_add(make_pmu_backend());      // uncore / NVML PCIe
            try_add(make_pcie_sysfs_backend());
            break;
    }

    if (synthetic_) {
        synthetic_->init(bindings_);
    }
    if (backends_.empty() && config_.allow_synthetic_fallback && synthetic_) {
        active_name_ = synthetic_->name();
    } else if (!backends_.empty()) {
        active_name_ = backends_.front()->name();
        for (size_t i = 1; i < backends_.size(); ++i) {
            active_name_ += std::string("+") + backends_[i]->name();
        }
        if (config_.allow_synthetic_fallback) active_name_ += "+synthetic";
    }
}

void TelemetryDaemon::start() {
    if (bindings_.empty()) {
        std::vector<FabricType> fabrics(num_edges_, FabricType::InfiniBand);
        std::vector<double> bw(num_edges_, 50.0);
        configure_edges(fabrics, bw);
    }
    if (running_.exchange(true)) return;
    collector_thread_ = std::thread([this] { collect_loop(); });
}

void TelemetryDaemon::stop() {
    if (!running_.exchange(false)) return;
    if (collector_thread_.joinable()) collector_thread_.join();
}

void TelemetryDaemon::set_simulated_util(uint32_t edge_idx, float util) {
    std::lock_guard<std::mutex> lock(mu_);
    if (edge_idx >= override_utils_.size()) return;
    override_utils_[edge_idx] = std::clamp(util, 0.0f, 1.0f);
    override_valid_[edge_idx] = 1;
}

const char* TelemetryDaemon::active_backend_name() const {
    return active_name_.c_str();
}

float TelemetryDaemon::read_edge_util(uint32_t edge_idx) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (edge_idx < override_valid_.size() && override_valid_[edge_idx]) {
            return override_utils_[edge_idx];
        }
    }

    FabricType fabric = FabricType::InfiniBand;
    if (edge_idx < bindings_.size()) fabric = bindings_[edge_idx].fabric;

    float util = 0.0f;
    bool got = false;
    for (auto& b : backends_) {
        // Prefer fabric-matched backends
        const char* n = b->name();
        const bool match =
            (fabric == FabricType::NVLink && std::strcmp(n, "nvml") == 0) ||
            (fabric == FabricType::InfiniBand &&
             (std::strcmp(n, "ib_mad") == 0 || std::strcmp(n, "ib_sysfs") == 0 ||
              std::strcmp(n, "ibverbs_sysfs") == 0)) ||
            ((fabric == FabricType::PCIe || fabric == FabricType::CXL) &&
             (std::strcmp(n, "perf_uncore") == 0 || std::strcmp(n, "sysfs_uncore") == 0 ||
              std::strcmp(n, "nvml_pcie") == 0 || std::strcmp(n, "pmu") == 0 ||
              std::strcmp(n, "pcie_sysfs") == 0 || std::strcmp(n, "nvml") == 0));
        if (!match && backends_.size() > 1) continue;
        if (b->sample(edge_idx, util)) {
            got = true;
            break;
        }
    }
    if (!got) {
        for (auto& b : backends_) {
            if (b->sample(edge_idx, util)) {
                got = true;
                break;
            }
        }
    }
    if (!got && config_.allow_synthetic_fallback && synthetic_) {
        synthetic_->sample(edge_idx, util);
    }
    return util;
}

void TelemetryDaemon::collect_loop() {
    using namespace std::chrono;
    const uint32_t hz = std::max(1u, config_.sample_hz);
    const auto interval = microseconds(1000000 / hz);
    while (running_.load(std::memory_order_relaxed)) {
        for (size_t e = 0; e < num_edges_; ++e) {
            float util = read_edge_util(static_cast<uint32_t>(e));
            ring_.write(static_cast<uint32_t>(e), util);
        }
        std::this_thread::sleep_for(interval);
    }
}

}  // namespace hics
