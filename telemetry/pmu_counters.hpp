#pragma once

#include "telemetry/hw_counters.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hics {

// Fine-grained interconnect PMU sampling:
//  - Linux perf_event_open on uncore IIO / IMC event sources
//  - sysfs uncore counter fallbacks
//  - NVML PCIe throughput when GPU-attached
struct PmuSample {
    double read_bw_gbps{0.0};
    double write_bw_gbps{0.0};
    double util{0.0};
    bool valid{false};
};

class PmuCounterClient {
public:
    PmuCounterClient();
    ~PmuCounterClient();

    bool available() const { return ready_; }
    const char* backend_name() const { return backend_name_.c_str(); }

    // Bind to a PCI device sysfs path and/or GPU index.
    bool open(const EdgeHwBinding& binding);
    void close();

    bool sample(PmuSample& out);

private:
    bool ready_{false};
    std::string backend_name_{"none"};
    EdgeHwBinding binding_{};

    // perf_event fds
    int fd_read_{-1};
    int fd_write_{-1};
    uint64_t prev_read_{0};
    uint64_t prev_write_{0};
    uint64_t prev_ts_ns_{0};

    // sysfs uncore paths
    std::string uncore_read_path_;
    std::string uncore_write_path_;

#if defined(__linux__)
    void* nvml_{nullptr};
    int (*nvmlInit_)(){nullptr};
    int (*nvmlDeviceGetHandleByIndex_)(unsigned, void**){nullptr};
    int (*nvmlDeviceGetPcieThroughput_)(void*, unsigned, unsigned int*){nullptr};
#endif

    bool open_perf_uncore();
    bool open_sysfs_uncore();
    bool open_nvml_pcie();
    bool sample_perf(PmuSample& out);
    bool sample_sysfs(PmuSample& out);
    bool sample_nvml(PmuSample& out);
};

std::unique_ptr<HwCounterBackend> make_pmu_backend();

}  // namespace hics
