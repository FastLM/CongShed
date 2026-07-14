#include "telemetry/pmu_counters.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
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

#if defined(__linux__)
int perf_event_open_syscall(struct perf_event_attr* attr, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return static_cast<int>(syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags));
}

// Discover first uncore_iio_* PMU type id from sysfs
bool find_uncore_iio_type(uint32_t& type_id, std::string& pmu_name) {
    DIR* dir = opendir("/sys/bus/event_source/devices");
    if (!dir) return false;
    while (auto* ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (name.rfind("uncore_iio", 0) != 0 && name.rfind("uncore_imc", 0) != 0)
            continue;
        std::string type_path =
            std::string("/sys/bus/event_source/devices/") + name + "/type";
        uint64_t t = 0;
        if (!read_u64_file(type_path, t)) continue;
        type_id = static_cast<uint32_t>(t);
        pmu_name = name;
        closedir(dir);
        return true;
    }
    closedir(dir);
    return false;
}

// Parse event format; many IIO PMUs expose data_read / data_write
bool find_uncore_event_config(const std::string& pmu, const char* event_name,
                              uint64_t& config) {
    std::string path =
        "/sys/bus/event_source/devices/" + pmu + "/events/" + event_name;
    std::ifstream f(path);
    if (!f) return false;
    // Format typically: "event=0xXX" or "event=0xXX,umask=0xYY"
    std::string line;
    std::getline(f, line);
    uint64_t event = 0, umask = 0;
    unsigned long long ev = 0, um = 0;
    if (std::sscanf(line.c_str(), "event=%llx", &ev) >= 1 ||
        std::sscanf(line.c_str(), "event=0x%llx", &ev) >= 1) {
        event = ev;
        auto pos = line.find("umask=");
        if (pos != std::string::npos) {
            if (std::sscanf(line.c_str() + pos, "umask=%llx", &um) >= 1 ||
                std::sscanf(line.c_str() + pos, "umask=0x%llx", &um) >= 1) {
                umask = um;
            }
        }
        config = event | (umask << 8);
        return true;
    }
    unsigned long long raw = 0;
    if (std::sscanf(line.c_str(), "%llx", &raw) == 1) {
        config = raw;
        return true;
    }
    return false;
}
#endif

}  // namespace

PmuCounterClient::PmuCounterClient() = default;

PmuCounterClient::~PmuCounterClient() { close(); }

bool PmuCounterClient::open(const EdgeHwBinding& binding) {
    close();
    binding_ = binding;

    if (open_perf_uncore()) {
        backend_name_ = "perf_uncore";
        ready_ = true;
        return true;
    }
    if (open_sysfs_uncore()) {
        backend_name_ = "sysfs_uncore";
        ready_ = true;
        return true;
    }
    if (open_nvml_pcie()) {
        backend_name_ = "nvml_pcie";
        ready_ = true;
        return true;
    }
    ready_ = false;
    return false;
}

void PmuCounterClient::close() {
#if defined(__linux__)
    if (fd_read_ >= 0) {
        ::close(fd_read_);
        fd_read_ = -1;
    }
    if (fd_write_ >= 0) {
        ::close(fd_write_);
        fd_write_ = -1;
    }
    if (nvml_) {
        dlclose(nvml_);
        nvml_ = nullptr;
    }
#endif
    ready_ = false;
}

bool PmuCounterClient::open_perf_uncore() {
#if defined(__linux__)
    uint32_t type = 0;
    std::string pmu;
    if (!find_uncore_iio_type(type, pmu)) return false;

    uint64_t cfg_r = 0, cfg_w = 0;
    // Common event names across Ice Lake / Sapphire Rapids IIO
    const char* read_names[] = {"data_read", "read_inserts", "event_data_read", nullptr};
    const char* write_names[] = {"data_write", "write_inserts", "event_data_write",
                                 nullptr};
    bool got_r = false, got_w = false;
    for (int i = 0; read_names[i]; ++i) {
        if (find_uncore_event_config(pmu, read_names[i], cfg_r)) {
            got_r = true;
            break;
        }
    }
    for (int i = 0; write_names[i]; ++i) {
        if (find_uncore_event_config(pmu, write_names[i], cfg_w)) {
            got_w = true;
            break;
        }
    }
    if (!got_r && !got_w) {
        // Generic uncore cycle/event fallback — still useful as activity proxy
        cfg_r = 0x01;
        cfg_w = 0x02;
        got_r = got_w = true;
    }

    auto open_one = [&](uint64_t config) -> int {
        perf_event_attr attr{};
        std::memset(&attr, 0, sizeof(attr));
        attr.type = type;
        attr.size = sizeof(attr);
        attr.config = config;
        attr.disabled = 1;
        attr.exclude_kernel = 0;
        attr.exclude_hv = 0;
        // cpu 0 / -1 pid = system-wide for uncore
        int fd = perf_event_open_syscall(&attr, -1, 0, -1, 0);
        if (fd < 0) return -1;
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        return fd;
    };

    if (got_r) fd_read_ = open_one(cfg_r);
    if (got_w) fd_write_ = open_one(cfg_w);
    return fd_read_ >= 0 || fd_write_ >= 0;
#else
    return false;
#endif
}

bool PmuCounterClient::open_sysfs_uncore() {
#if defined(__linux__)
    // Intel PCM / kernel uncore often exposes:
    //   /sys/devices/uncore_iio_0/counters or debugfs
    // Also try PCI device bandwidth estimate files.
    if (!binding_.pci_sysfs.empty()) {
        // Some platforms export `mem_bandwidth` style debug counters under device
        const std::string candidates[] = {
            binding_.pci_sysfs + "/mem_read_bandwidth",
            binding_.pci_sysfs + "/mem_write_bandwidth",
        };
        (void)candidates;
    }

    DIR* dir = opendir("/sys/bus/event_source/devices");
    if (!dir) return false;
    while (auto* ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (name.rfind("uncore_", 0) != 0) continue;
        // Look for a readable "cpumask" as presence check
        std::string base = std::string("/sys/bus/event_source/devices/") + name;
        std::ifstream cpumask(base + "/cpumask");
        if (!cpumask) continue;
        // Store paths to events if present for delta sampling via
        // /sys/devices/... not always directly readable as counters —
        // keep PCI link speed/width as capacity; activity from
        // `/sys/class/drm` or NVML later.
        uncore_read_path_ = base + "/events/data_read";
        uncore_write_path_ = base + "/events/data_write";
        if (std::ifstream(uncore_read_path_) || std::ifstream(uncore_write_path_)) {
            closedir(dir);
            // Event description files alone aren't counters; signal soft-available
            // so NVML / perf paths preferred. Return false to continue probing.
            break;
        }
    }
    closedir(dir);

    // PCIe link training info as last-resort capacity occupancy proxy:
    // combine with AER correctable error rate if present.
    if (!binding_.pci_sysfs.empty()) {
        uint64_t width = 0;
        if (read_u64_file(binding_.pci_sysfs + "/current_link_width", width) &&
            width > 0) {
            // Mark open — sample() will compute from AER + link
            uncore_read_path_ = binding_.pci_sysfs + "/aer_dev_correctable";
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

bool PmuCounterClient::open_nvml_pcie() {
#if defined(__linux__)
    if (binding_.gpu_a < 0) return false;
    nvml_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!nvml_) nvml_ = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
    if (!nvml_) return false;
    nvmlInit_ = reinterpret_cast<int (*)()>(dlsym(nvml_, "nvmlInit_v2"));
    if (!nvmlInit_) nvmlInit_ = reinterpret_cast<int (*)()>(dlsym(nvml_, "nvmlInit"));
    nvmlDeviceGetHandleByIndex_ = reinterpret_cast<int (*)(unsigned, void**)>(
        dlsym(nvml_, "nvmlDeviceGetHandleByIndex_v2"));
    if (!nvmlDeviceGetHandleByIndex_)
        nvmlDeviceGetHandleByIndex_ = reinterpret_cast<int (*)(unsigned, void**)>(
            dlsym(nvml_, "nvmlDeviceGetHandleByIndex"));
    nvmlDeviceGetPcieThroughput_ =
        reinterpret_cast<int (*)(void*, unsigned, unsigned int*)>(
            dlsym(nvml_, "nvmlDeviceGetPcieThroughput"));
    if (!nvmlInit_ || !nvmlDeviceGetHandleByIndex_ || !nvmlDeviceGetPcieThroughput_) {
        dlclose(nvml_);
        nvml_ = nullptr;
        return false;
    }
    if (nvmlInit_() != 0) {
        dlclose(nvml_);
        nvml_ = nullptr;
        return false;
    }
    return true;
#else
    return false;
#endif
}

bool PmuCounterClient::sample_perf(PmuSample& out) {
#if defined(__linux__)
    auto read_fd = [](int fd, uint64_t& val) -> bool {
        if (fd < 0) return false;
        long long v = 0;
        if (::read(fd, &v, sizeof(v)) != sizeof(v)) return false;
        val = static_cast<uint64_t>(v);
        return true;
    };
    uint64_t r = 0, w = 0;
    const bool got_r = read_fd(fd_read_, r);
    const bool got_w = read_fd(fd_write_, w);
    if (!got_r && !got_w) return false;

    const uint64_t now = steady_ns();
    if (prev_ts_ns_ == 0) {
        prev_read_ = r;
        prev_write_ = w;
        prev_ts_ns_ = now;
        out = {};
        out.valid = true;
        return true;
    }
    const double dt = (now - prev_ts_ns_) * 1e-9;
    // Uncore IIO events are often cacheline (64B) inserts
    constexpr double kBytesPerCount = 64.0;
    const double rb = got_r ? (r - prev_read_) * kBytesPerCount : 0.0;
    const double wb = got_w ? (w - prev_write_) * kBytesPerCount : 0.0;
    out.read_bw_gbps = rb / (dt * 1e9 + 1e-12);
    out.write_bw_gbps = wb / (dt * 1e9 + 1e-12);
    const double peak = std::max(binding_.peak_bw_gbps, 1.0);
    out.util = std::clamp((out.read_bw_gbps + out.write_bw_gbps) / peak, 0.0, 0.99);
    out.valid = true;
    prev_read_ = r;
    prev_write_ = w;
    prev_ts_ns_ = now;
    return true;
#else
    (void)out;
    return false;
#endif
}

bool PmuCounterClient::sample_sysfs(PmuSample& out) {
#if defined(__linux__)
    out = {};
    // AER correctable error counter growth as congestion/noise proxy + link width
    uint64_t width = 0;
    read_u64_file(binding_.pci_sysfs + "/current_link_width", width);

    // Parse speed string like "16.0 GT/s PCIe"
    double gt_s = 16.0;
    {
        std::ifstream sf(binding_.pci_sysfs + "/current_link_speed");
        std::string s;
        if (sf && std::getline(sf, s)) {
            double v = 0;
            if (std::sscanf(s.c_str(), "%lf", &v) == 1) gt_s = v;
        }
    }
    // PCIe payload ≈ GT/s * width * encoding efficiency (0.98 for Gen5)
    const double peak = (gt_s * std::max<uint64_t>(width, 1) * 0.98) / 8.0;  // GB/s

    uint64_t aer = 0;
    const bool got_aer = !uncore_read_path_.empty() && read_u64_file(uncore_read_path_, aer);
    const uint64_t now = steady_ns();
    if (prev_ts_ns_ == 0) {
        prev_read_ = aer;
        prev_ts_ns_ = now;
        out.util = 0.05f;
        out.valid = true;
        return true;
    }
    double util = 0.05;
    if (got_aer && aer >= prev_read_) {
        const double dt = (now - prev_ts_ns_) * 1e-9;
        const double rate = (aer - prev_read_) / (dt + 1e-12);
        util = std::clamp(0.05 + rate * 1e-3, 0.0, 0.99);
        prev_read_ = aer;
    }
    // Scale reported util relative to negotiated link capacity vs binding peak
    if (peak > 0 && binding_.peak_bw_gbps > 0) {
        util = std::min(0.99, util * (binding_.peak_bw_gbps / peak));
    }
    out.util = util;
    out.read_bw_gbps = util * binding_.peak_bw_gbps * 0.5;
    out.write_bw_gbps = util * binding_.peak_bw_gbps * 0.5;
    out.valid = true;
    prev_ts_ns_ = now;
    return true;
#else
    (void)out;
    return false;
#endif
}

bool PmuCounterClient::sample_nvml(PmuSample& out) {
#if defined(__linux__)
    if (!nvmlDeviceGetHandleByIndex_ || !nvmlDeviceGetPcieThroughput_) return false;
    void* dev = nullptr;
    if (nvmlDeviceGetHandleByIndex_(static_cast<unsigned>(binding_.gpu_a), &dev) != 0)
        return false;
    unsigned int tx_kbps = 0, rx_kbps = 0;
    // 0 = TX, 1 = RX
    if (nvmlDeviceGetPcieThroughput_(dev, 0, &tx_kbps) != 0) return false;
    nvmlDeviceGetPcieThroughput_(dev, 1, &rx_kbps);
    out.write_bw_gbps = static_cast<double>(tx_kbps) / (1024.0 * 1024.0);
    out.read_bw_gbps = static_cast<double>(rx_kbps) / (1024.0 * 1024.0);
    const double peak = std::max(binding_.peak_bw_gbps, 1.0);
    out.util =
        std::clamp((out.read_bw_gbps + out.write_bw_gbps) / peak, 0.0, 0.99);
    out.valid = true;
    return true;
#else
    (void)out;
    return false;
#endif
}

bool PmuCounterClient::sample(PmuSample& out) {
    if (!ready_) return false;
    if (fd_read_ >= 0 || fd_write_ >= 0) return sample_perf(out);
#if defined(__linux__)
    if (nvml_) return sample_nvml(out);
#endif
    return sample_sysfs(out);
}

class PmuBackend final : public HwCounterBackend {
public:
    const char* name() const override { return name_.c_str(); }

    bool available() const override {
#if defined(__linux__)
        return true;  // probe in init
#else
        return false;
#endif
    }

    bool init(const std::vector<EdgeHwBinding>& bindings) override {
        bindings_ = bindings;
        clients_.clear();
        clients_.resize(bindings.size());
        bool any = false;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (bindings[i].fabric != FabricType::PCIe &&
                bindings[i].fabric != FabricType::CXL) {
                continue;
            }
            auto c = std::make_unique<PmuCounterClient>();
            if (!c->open(bindings[i])) continue;
            name_ = c->backend_name();
            clients_[i] = std::move(c);
            any = true;
        }
        return any;
    }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (edge_idx >= clients_.size() || !clients_[edge_idx]) return false;
        PmuSample s;
        if (!clients_[edge_idx]->sample(s) || !s.valid) return false;
        out_util = static_cast<float>(s.util);
        return true;
    }

private:
    std::vector<EdgeHwBinding> bindings_;
    std::vector<std::unique_ptr<PmuCounterClient>> clients_;
    std::string name_{"pmu"};
};

std::unique_ptr<HwCounterBackend> make_pmu_backend() {
    return std::make_unique<PmuBackend>();
}

}  // namespace hics
