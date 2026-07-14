#pragma once

#include "model/model.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hics {

// Per-edge hardware binding used by counter backends
struct EdgeHwBinding {
    uint32_t edge_idx{0};
    FabricType fabric{FabricType::InfiniBand};
    int gpu_a{-1};           // NVML device index (NVLink / PCIe)
    int gpu_b{-1};
    int nvlink_id{-1};       // NVLink lane id, or -1
    std::string ib_device;   // e.g. "mlx5_0"
    int ib_port{1};
    std::string pci_sysfs;   // e.g. "/sys/bus/pci/devices/0000:00:00.0"
    double peak_bw_gbps{50.0};
};

enum class TelemetryBackendKind : uint8_t {
    Auto = 0,
    Nvml = 1,
    Ibverbs = 2,
    PcieSysfs = 3,
    Synthetic = 4,
    IbMad = 5,
    Pmu = 6,
};

class HwCounterBackend {
public:
    virtual ~HwCounterBackend() = default;
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual bool init(const std::vector<EdgeHwBinding>& bindings) = 0;
    // Returns utilization ∈ [0,1] for the edge; false if sample unavailable
    virtual bool sample(uint32_t edge_idx, float& out_util) = 0;
};

std::unique_ptr<HwCounterBackend> make_nvml_backend();
std::unique_ptr<HwCounterBackend> make_ibverbs_backend();
std::unique_ptr<HwCounterBackend> make_pcie_sysfs_backend();
std::unique_ptr<HwCounterBackend> make_synthetic_backend();
std::unique_ptr<HwCounterBackend> make_ib_mad_backend();
std::unique_ptr<HwCounterBackend> make_pmu_backend();

// Build default bindings from topology fabric attributes
std::vector<EdgeHwBinding> bindings_from_fabrics(
    const std::vector<FabricType>& fabrics,
    const std::vector<double>& peak_bw_gbps);

}  // namespace hics
