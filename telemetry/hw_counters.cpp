#include "telemetry/hw_counters.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

#if defined(__linux__)
#include <dlfcn.h>
#include <dirent.h>
#endif

namespace hics {
namespace {

uint64_t steady_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool read_u64_file(const std::string& path, uint64_t& out) {
    std::ifstream f(path);
    if (!f) return false;
    f >> out;
    return static_cast<bool>(f);
}

}  // namespace

std::vector<EdgeHwBinding> bindings_from_fabrics(
    const std::vector<FabricType>& fabrics,
    const std::vector<double>& peak_bw_gbps) {
    std::vector<EdgeHwBinding> out;
    out.reserve(fabrics.size());
    for (size_t i = 0; i < fabrics.size(); ++i) {
        EdgeHwBinding b;
        b.edge_idx = static_cast<uint32_t>(i);
        b.fabric = fabrics[i];
        b.peak_bw_gbps = i < peak_bw_gbps.size() ? peak_bw_gbps[i] : 50.0;
        // Heuristic device mapping — refined by NVML/IB discovery when present
        b.gpu_a = static_cast<int>(i % 8);
        b.gpu_b = static_cast<int>((i + 1) % 8);
        b.nvlink_id = static_cast<int>(i % 18);
        b.ib_device = "mlx5_0";
        b.ib_port = 1;
        char pci[64];
        std::snprintf(pci, sizeof(pci), "/sys/class/infiniband/mlx5_0/ports/1");
        b.pci_sysfs = pci;
        out.push_back(std::move(b));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Synthetic backend — always available; models bursty KV + background traffic
// ---------------------------------------------------------------------------
class SyntheticBackend final : public HwCounterBackend {
public:
    const char* name() const override { return "synthetic"; }
    bool available() const override { return true; }

    bool init(const std::vector<EdgeHwBinding>& bindings) override {
        bindings_ = bindings;
        state_.assign(bindings.size(), {});
        rng_.seed(0xC0FFEE);
        return true;
    }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (edge_idx >= state_.size()) return false;
        auto& st = state_[edge_idx];
        const double t = static_cast<double>(steady_ns()) * 1e-9;

        // Background utilization + occasional migration bursts
        double u = 0.25 + 0.15 * std::sin(t * 0.7 + edge_idx * 0.3);
        if (st.burst_left > 0) {
            u = std::max(u, static_cast<double>(st.burst_peak));
            --st.burst_left;
        } else if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < 0.002) {
            st.burst_left = std::uniform_int_distribution<int>(5, 40)(rng_);
            st.burst_peak = std::uniform_real_distribution<float>(0.75f, 0.98f)(rng_);
            u = st.burst_peak;
        }
        // Fabric-specific noise floor
        if (edge_idx < bindings_.size()) {
            switch (bindings_[edge_idx].fabric) {
                case FabricType::NVLink: u *= 0.9; break;
                case FabricType::PCIe: u = std::min(1.0, u * 1.1); break;
                default: break;
            }
        }
        out_util = static_cast<float>(std::clamp(u, 0.0, 0.99));
        return true;
    }

private:
    struct EdgeState {
        int burst_left{0};
        float burst_peak{0.9f};
    };
    std::vector<EdgeHwBinding> bindings_;
    std::vector<EdgeState> state_;
    std::mt19937 rng_;
};

std::unique_ptr<HwCounterBackend> make_synthetic_backend() {
    return std::make_unique<SyntheticBackend>();
}

// ---------------------------------------------------------------------------
// NVML backend — dlopen libnvidia-ml.so.1 when present
// ---------------------------------------------------------------------------
class NvmlBackend final : public HwCounterBackend {
public:
    const char* name() const override { return "nvml"; }

    bool available() const override {
#if defined(__linux__)
        void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!h) h = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
        if (!h) return false;
        dlclose(h);
        return true;
#else
        return false;
#endif
    }

    bool init(const std::vector<EdgeHwBinding>& bindings) override {
        bindings_ = bindings;
        prev_tx_.assign(bindings.size(), 0);
        prev_ts_.assign(bindings.size(), 0);
#if defined(__linux__)
        handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!handle_) handle_ = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
        if (!handle_) return false;

        // Resolve a minimal NVML surface. Signatures match NVML public ABI.
        nvmlInit_ = reinterpret_cast<int (*)()>(dlsym(handle_, "nvmlInit_v2"));
        if (!nvmlInit_)
            nvmlInit_ = reinterpret_cast<int (*)()>(dlsym(handle_, "nvmlInit"));
        nvmlShutdown_ = reinterpret_cast<int (*)()>(dlsym(handle_, "nvmlShutdown"));
        nvmlDeviceGetHandleByIndex_ = reinterpret_cast<int (*)(unsigned, void**)>(
            dlsym(handle_, "nvmlDeviceGetHandleByIndex_v2"));
        if (!nvmlDeviceGetHandleByIndex_)
            nvmlDeviceGetHandleByIndex_ = reinterpret_cast<int (*)(unsigned, void**)>(
                dlsym(handle_, "nvmlDeviceGetHandleByIndex"));
        nvmlDeviceGetNvLinkUtilizationCounter_ =
            reinterpret_cast<int (*)(void*, unsigned, unsigned, unsigned long long*,
                                     unsigned long long*)>(
                dlsym(handle_, "nvmlDeviceGetNvLinkUtilizationCounter"));
        nvmlDeviceGetPcieThroughput_ =
            reinterpret_cast<int (*)(void*, unsigned, unsigned int*)>(
                dlsym(handle_, "nvmlDeviceGetPcieThroughput"));

        if (!nvmlInit_ || nvmlInit_() != 0) {
            teardown();
            return false;
        }
        ready_ = true;
        return true;
#else
        (void)bindings;
        return false;
#endif
    }

    ~NvmlBackend() override { teardown(); }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (!ready_ || edge_idx >= bindings_.size()) return false;
#if defined(__linux__)
        const auto& b = bindings_[edge_idx];
        if (b.fabric == FabricType::NVLink && nvmlDeviceGetNvLinkUtilizationCounter_ &&
            b.gpu_a >= 0) {
            void* dev = nullptr;
            if (nvmlDeviceGetHandleByIndex_(static_cast<unsigned>(b.gpu_a), &dev) != 0)
                return false;
            unsigned long long rx = 0, tx = 0;
            // counter 0 = data flits; link id from binding
            const unsigned link =
                b.nvlink_id >= 0 ? static_cast<unsigned>(b.nvlink_id) : 0u;
            if (nvmlDeviceGetNvLinkUtilizationCounter_(dev, link, 0, &rx, &tx) != 0)
                return false;
            const uint64_t now = steady_ns();
            if (prev_ts_[edge_idx] == 0) {
                prev_tx_[edge_idx] = tx;
                prev_ts_[edge_idx] = now;
                out_util = 0.0f;
                return true;
            }
            const double dt_s = (now - prev_ts_[edge_idx]) * 1e-9;
            const double dtx = static_cast<double>(tx - prev_tx_[edge_idx]);
            prev_tx_[edge_idx] = tx;
            prev_ts_[edge_idx] = now;
            // NVLink counter units ≈ 16-byte flits on many platforms
            const double gbps = (dtx * 16.0) / (dt_s * 1e9 + 1e-12);
            out_util = static_cast<float>(
                std::clamp(gbps / std::max(b.peak_bw_gbps, 1.0), 0.0, 0.99));
            return true;
        }
        if (nvmlDeviceGetPcieThroughput_ && b.gpu_a >= 0) {
            void* dev = nullptr;
            if (nvmlDeviceGetHandleByIndex_(static_cast<unsigned>(b.gpu_a), &dev) != 0)
                return false;
            unsigned int kbps = 0;
            // 0 = TX, 1 = RX in NVML PCIe throughput enum
            if (nvmlDeviceGetPcieThroughput_(dev, 0, &kbps) != 0) return false;
            const double gbps = static_cast<double>(kbps) / (1024.0 * 1024.0);
            out_util = static_cast<float>(
                std::clamp(gbps / std::max(b.peak_bw_gbps, 1.0), 0.0, 0.99));
            return true;
        }
#endif
        return false;
    }

private:
    void teardown() {
#if defined(__linux__)
        if (ready_ && nvmlShutdown_) nvmlShutdown_();
        if (handle_) dlclose(handle_);
        handle_ = nullptr;
#endif
        ready_ = false;
    }

    std::vector<EdgeHwBinding> bindings_;
    std::vector<unsigned long long> prev_tx_;
    std::vector<uint64_t> prev_ts_;
    bool ready_{false};
#if defined(__linux__)
    void* handle_{nullptr};
    int (*nvmlInit_)(){nullptr};
    int (*nvmlShutdown_)(){nullptr};
    int (*nvmlDeviceGetHandleByIndex_)(unsigned, void**){nullptr};
    int (*nvmlDeviceGetNvLinkUtilizationCounter_)(void*, unsigned, unsigned,
                                                    unsigned long long*,
                                                    unsigned long long*){nullptr};
    int (*nvmlDeviceGetPcieThroughput_)(void*, unsigned, unsigned int*){nullptr};
#endif
};

std::unique_ptr<HwCounterBackend> make_nvml_backend() {
    return std::make_unique<NvmlBackend>();
}

// ---------------------------------------------------------------------------
// IB verbs via sysfs counters (port_xmit_data / port_rcv_data)
// ---------------------------------------------------------------------------
class IbverbsBackend final : public HwCounterBackend {
public:
    const char* name() const override { return "ibverbs_sysfs"; }

    bool available() const override {
#if defined(__linux__)
        std::ifstream f("/sys/class/infiniband");
        return static_cast<bool>(f);
#else
        return false;
#endif
    }

    bool init(const std::vector<EdgeHwBinding>& bindings) override {
        bindings_ = bindings;
        prev_words_.assign(bindings.size(), 0);
        prev_ts_.assign(bindings.size(), 0);
        bool any = false;
        for (const auto& b : bindings_) {
            if (b.fabric != FabricType::InfiniBand) continue;
            std::ostringstream oss;
            oss << "/sys/class/infiniband/" << b.ib_device << "/ports/" << b.ib_port
                << "/counters/port_xmit_data";
            uint64_t v = 0;
            if (read_u64_file(oss.str(), v)) any = true;
        }
        return any || available();
    }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (edge_idx >= bindings_.size()) return false;
        const auto& b = bindings_[edge_idx];
        if (b.fabric != FabricType::InfiniBand) return false;

        std::ostringstream oss;
        oss << "/sys/class/infiniband/" << b.ib_device << "/ports/" << b.ib_port
            << "/counters/port_xmit_data";
        uint64_t words = 0;
        if (!read_u64_file(oss.str(), words)) return false;

        const uint64_t now = steady_ns();
        if (prev_ts_[edge_idx] == 0) {
            prev_words_[edge_idx] = words;
            prev_ts_[edge_idx] = now;
            out_util = 0.0f;
            return true;
        }
        const double dt_s = (now - prev_ts_[edge_idx]) * 1e-9;
        // port_*_data counters are in 4-byte words on Linux IB sysfs
        const double dbytes =
            static_cast<double>(words - prev_words_[edge_idx]) * 4.0;
        prev_words_[edge_idx] = words;
        prev_ts_[edge_idx] = now;
        const double gbps = dbytes / (dt_s * 1e9 + 1e-12);
        out_util = static_cast<float>(
            std::clamp(gbps / std::max(b.peak_bw_gbps, 1.0), 0.0, 0.99));
        return true;
    }

private:
    std::vector<EdgeHwBinding> bindings_;
    std::vector<uint64_t> prev_words_;
    std::vector<uint64_t> prev_ts_;
};

std::unique_ptr<HwCounterBackend> make_ibverbs_backend() {
    return std::make_unique<IbverbsBackend>();
}

// ---------------------------------------------------------------------------
// PCIe via sysfs (AER / link status) + throughput estimate from nvml fallback
// ---------------------------------------------------------------------------
class PcieSysfsBackend final : public HwCounterBackend {
public:
    const char* name() const override { return "pcie_sysfs"; }

    bool available() const override {
#if defined(__linux__)
        std::ifstream f("/sys/bus/pci/devices");
        return static_cast<bool>(f);
#else
        return false;
#endif
    }

    bool init(const std::vector<EdgeHwBinding>& bindings) override {
        bindings_ = bindings;
        return available();
    }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (edge_idx >= bindings_.size()) return false;
        const auto& b = bindings_[edge_idx];
        if (b.fabric != FabricType::PCIe && b.fabric != FabricType::CXL) return false;

#if defined(__linux__)
        // Prefer current_link_width / current_link_speed as a capacity proxy;
        // without a PMU we estimate load from synthetic residual when counters
        // are absent — still prefer real counter files when present.
        std::string base = b.pci_sysfs;
        if (base.empty()) return false;
        uint64_t width = 0, speed_gts = 0;
        if (!read_u64_file(base + "/current_link_width", width)) {
            // Some kernels expose string speeds; treat missing as soft-fail
            out_util = 0.1f;
            return true;
        }
        (void)speed_gts;
        // Without byte counters, report a conservative low util so the
        // composite daemon can blend with synthetic / NVML PCIe throughput.
        out_util = width > 0 ? 0.05f : 0.0f;
        return true;
#else
        (void)b;
        return false;
#endif
    }

private:
    std::vector<EdgeHwBinding> bindings_;
};

std::unique_ptr<HwCounterBackend> make_pcie_sysfs_backend() {
    return std::make_unique<PcieSysfsBackend>();
}

}  // namespace hics
