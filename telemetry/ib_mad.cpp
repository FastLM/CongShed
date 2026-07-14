#include "telemetry/ib_mad.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#if defined(__linux__)
#include <arpa/inet.h>
#include <dlfcn.h>
#include <endian.h>
#include <unistd.h>
#endif

namespace hics {
namespace {

uint64_t steady_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool read_u64(const std::string& path, uint64_t& out) {
    std::ifstream f(path);
    if (!f) return false;
    f >> out;
    return static_cast<bool>(f);
}

// IBTA PerfMgt class / attributes
constexpr int kIbMadClassPerf = 0x04;
constexpr int kIbMadMethodGet = 0x01;
constexpr int kIbMadAttrPortCounters = 0x12;
constexpr int kIbMadAttrPortCountersExt = 0x1D;
constexpr int kMadHdrSize = 24;
constexpr int kUmadTimeoutMs = 20;

#pragma pack(push, 1)
struct MadHeader {
    uint8_t base_version{1};
    uint8_t mgmt_class{kIbMadClassPerf};
    uint8_t class_version{1};
    uint8_t method{kIbMadMethodGet};
    uint16_t status{0};
    uint16_t class_specific{0};
    uint64_t tid{0};
    uint16_t attr_id{0};
    uint16_t resv{0};
    uint32_t attr_mod{0};
};

struct PortCountersBody {
    uint8_t reserved0{0};
    uint8_t port_select{0};
    uint16_t counter_select{0xffff};
    uint16_t symbol_err{0};
    uint8_t link_err{0};
    uint8_t link_downed{0};
    uint8_t rcv_err{0};
    uint8_t rcv_rem_phys{0};
    uint8_t rcv_switch_relay{0};
    uint8_t xmit_discards{0};
    uint8_t xmit_constraint{0};
    uint8_t rcv_constraint{0};
    uint8_t reserved1{0};
    uint8_t link_integ{0};
    uint8_t buf_overrun{0};
    uint8_t vl15_dropped{0};
    uint32_t xmit_data{0};
    uint32_t rcv_data{0};
    uint32_t xmit_pkts{0};
    uint32_t rcv_pkts{0};
    uint32_t xmit_wait{0};
};

struct PortCountersExtBody {
    uint8_t reserved0{0};
    uint8_t port_select{0};
    uint16_t counter_select{0xffff};
    uint32_t reserved1{0};
    uint64_t xmit_data{0};
    uint64_t rcv_data{0};
    uint64_t xmit_pkts{0};
    uint64_t rcv_pkts{0};
    uint64_t unicast_xmit{0};
    uint64_t unicast_rcv{0};
    uint64_t multicast_xmit{0};
    uint64_t multicast_rcv{0};
};
#pragma pack(pop)

}  // namespace

IbMadClient::IbMadClient() { init_mad_libs(); }

IbMadClient::~IbMadClient() { close(); }

bool IbMadClient::init_mad_libs() {
#if defined(__linux__)
    umad_handle_ = dlopen("libibumad.so.3", RTLD_LAZY | RTLD_LOCAL);
    if (!umad_handle_) umad_handle_ = dlopen("libibumad.so", RTLD_LAZY | RTLD_LOCAL);
    mad_handle_ = dlopen("libibmad.so.5", RTLD_LAZY | RTLD_LOCAL);
    if (!mad_handle_) mad_handle_ = dlopen("libibmad.so", RTLD_LAZY | RTLD_LOCAL);

    if (umad_handle_) {
        umad_init_ = reinterpret_cast<int (*)()>(dlsym(umad_handle_, "umad_init"));
        umad_done_ = reinterpret_cast<int (*)()>(dlsym(umad_handle_, "umad_done"));
        umad_open_port_ = reinterpret_cast<int (*)(const char*, int)>(
            dlsym(umad_handle_, "umad_open_port"));
        umad_close_port_ =
            reinterpret_cast<int (*)(int)>(dlsym(umad_handle_, "umad_close_port"));
        umad_register_ = reinterpret_cast<int (*)(int, int, int, uint32_t, void*)>(
            dlsym(umad_handle_, "umad_register"));
        umad_unregister_ =
            reinterpret_cast<int (*)(int, int)>(dlsym(umad_handle_, "umad_unregister"));
        umad_alloc_ =
            reinterpret_cast<void* (*)(int, size_t)>(dlsym(umad_handle_, "umad_alloc"));
        umad_free_ = reinterpret_cast<void (*)(void*)>(dlsym(umad_handle_, "umad_free"));
        umad_send_ = reinterpret_cast<int (*)(int, int, void*, int, int, int)>(
            dlsym(umad_handle_, "umad_send"));
        umad_recv_ = reinterpret_cast<int (*)(int, void*, int*, int)>(
            dlsym(umad_handle_, "umad_recv"));
        umad_get_mad_ =
            reinterpret_cast<void* (*)(void*)>(dlsym(umad_handle_, "umad_get_mad"));
        umad_size_ = reinterpret_cast<size_t (*)()>(dlsym(umad_handle_, "umad_size"));
    }

    if (umad_init_ && umad_open_port_ && umad_register_ && umad_send_ && umad_recv_ &&
        umad_alloc_ && umad_get_mad_) {
        if (umad_init_() == 0) {
            use_mad_ = true;
            backend_name_ = "ib_mad";
            return true;
        }
    }
    // Sysfs still usable without MAD libs
    backend_name_ = "ib_sysfs";
    ready_ = true;  // open() will verify device paths
    return true;
#else
    ready_ = false;
    return false;
#endif
}

bool IbMadClient::open(const std::string& ca_name, int port) {
    close();
    ca_name_ = ca_name;
    port_ = port;
    std::ostringstream oss;
    oss << "/sys/class/infiniband/" << ca_name << "/ports/" << port;
    sysfs_base_ = oss.str();

#if defined(__linux__)
    if (use_mad_ && umad_open_port_) {
        umad_fd_ = umad_open_port_(ca_name.c_str(), port);
        if (umad_fd_ >= 0) {
            // Register PerfMgt class agent (mgmt_class=4, class_version=1)
            umad_agent_ = umad_register_(umad_fd_, kIbMadClassPerf, 1, 0, nullptr);
            if (umad_agent_ >= 0) {
                backend_name_ = "ib_mad";
                ready_ = true;
                return true;
            }
            if (umad_close_port_) umad_close_port_(umad_fd_);
            umad_fd_ = -1;
        }
    }
#endif
    // Sysfs fallback
    uint64_t probe = 0;
    if (read_u64(sysfs_base_ + "/counters/port_xmit_data", probe) ||
        read_u64(sysfs_base_ + "/counters_ext/port_xmit_data", probe)) {
        backend_name_ = "ib_sysfs";
        ready_ = true;
        return true;
    }
    ready_ = false;
    return false;
}

void IbMadClient::close() {
#if defined(__linux__)
    if (umad_fd_ >= 0) {
        if (umad_agent_ >= 0 && umad_unregister_)
            umad_unregister_(umad_fd_, umad_agent_);
        if (umad_close_port_) umad_close_port_(umad_fd_);
    }
    umad_fd_ = -1;
    umad_agent_ = -1;
#endif
}

bool IbMadClient::query_via_sysfs(IbPortCounters& out) {
    out = {};
    // Prefer 64-bit extended counters when present
    if (read_u64(sysfs_base_ + "/counters_ext/port_xmit_data", out.port_xmit_data) &&
        read_u64(sysfs_base_ + "/counters_ext/port_rcv_data", out.port_rcv_data)) {
        read_u64(sysfs_base_ + "/counters_ext/port_xmit_packets", out.port_xmit_pkts);
        read_u64(sysfs_base_ + "/counters_ext/port_rcv_packets", out.port_rcv_pkts);
        out.extended = true;
        out.valid = true;
    } else if (read_u64(sysfs_base_ + "/counters/port_xmit_data", out.port_xmit_data) &&
               read_u64(sysfs_base_ + "/counters/port_rcv_data", out.port_rcv_data)) {
        read_u64(sysfs_base_ + "/counters/port_xmit_packets", out.port_xmit_pkts);
        read_u64(sysfs_base_ + "/counters/port_rcv_packets", out.port_rcv_pkts);
        read_u64(sysfs_base_ + "/counters/port_xmit_wait", out.xmit_wait);
        read_u64(sysfs_base_ + "/counters/port_xmit_discards", out.port_xmit_discards);
        read_u64(sysfs_base_ + "/counters/symbol_error", out.symbol_err);
        read_u64(sysfs_base_ + "/counters/link_error_recovery", out.link_err);
        out.extended = false;
        out.valid = true;
    }
    return out.valid;
}

bool IbMadClient::build_and_send_perfmgt(bool extended, IbPortCounters& out) {
#if defined(__linux__)
    if (umad_fd_ < 0 || !umad_alloc_ || !umad_send_ || !umad_recv_ || !umad_get_mad_)
        return false;

    const size_t body = extended ? sizeof(PortCountersExtBody) : sizeof(PortCountersBody);
    const size_t mad_len = kMadHdrSize + body;
    void* umad = umad_alloc_(1, static_cast<int>(mad_len));
    if (!umad) return false;

    void* mad = umad_get_mad_(umad);
    std::memset(mad, 0, mad_len);
    auto* hdr = reinterpret_cast<MadHeader*>(mad);
    hdr->base_version = 1;
    hdr->mgmt_class = kIbMadClassPerf;
    hdr->class_version = extended ? 2 : 1;
    hdr->method = kIbMadMethodGet;
    hdr->tid = steady_ns();
    hdr->attr_id =
        htons(static_cast<uint16_t>(extended ? kIbMadAttrPortCountersExt
                                             : kIbMadAttrPortCounters));

    if (extended) {
        auto* b = reinterpret_cast<PortCountersExtBody*>(
            reinterpret_cast<uint8_t*>(mad) + kMadHdrSize);
        b->port_select = static_cast<uint8_t>(port_);
        b->counter_select = htons(0xffff);
    } else {
        auto* b = reinterpret_cast<PortCountersBody*>(
            reinterpret_cast<uint8_t*>(mad) + kMadHdrSize);
        b->port_select = static_cast<uint8_t>(port_);
        b->counter_select = htons(0xffff);
    }

    // LID-routed to local port (lid 0 / directed route often used for local QP0)
    int length = static_cast<int>(mad_len);
    if (umad_send_(umad_fd_, umad_agent_, umad, length, kUmadTimeoutMs, 0) < 0) {
        umad_free_(umad);
        return false;
    }
    int recv_len = static_cast<int>(mad_len);
    if (umad_recv_(umad_fd_, umad, &recv_len, kUmadTimeoutMs) < 0) {
        umad_free_(umad);
        return false;
    }

    mad = umad_get_mad_(umad);
    hdr = reinterpret_cast<MadHeader*>(mad);
    if (hdr->status != 0) {
        umad_free_(umad);
        return false;
    }

    out = {};
    if (extended) {
        auto* b = reinterpret_cast<PortCountersExtBody*>(
            reinterpret_cast<uint8_t*>(mad) + kMadHdrSize);
        // Network byte order for multi-byte fields in MAD payloads
        out.port_xmit_data = be64toh(b->xmit_data);
        out.port_rcv_data = be64toh(b->rcv_data);
        out.port_xmit_pkts = be64toh(b->xmit_pkts);
        out.port_rcv_pkts = be64toh(b->rcv_pkts);
        out.extended = true;
    } else {
        auto* b = reinterpret_cast<PortCountersBody*>(
            reinterpret_cast<uint8_t*>(mad) + kMadHdrSize);
        out.port_xmit_data = ntohl(b->xmit_data);
        out.port_rcv_data = ntohl(b->rcv_data);
        out.port_xmit_pkts = ntohl(b->xmit_pkts);
        out.port_rcv_pkts = ntohl(b->rcv_pkts);
        out.xmit_wait = ntohl(b->xmit_wait);
        out.symbol_err = ntohs(b->symbol_err);
        out.link_err = b->link_err;
        out.port_xmit_discards = b->xmit_discards;
        out.extended = false;
    }
    out.valid = true;
    umad_free_(umad);
    return true;
#else
    (void)extended;
    (void)out;
    return false;
#endif
}

bool IbMadClient::query_via_mad(IbPortCounters& out) {
    if (build_and_send_perfmgt(true, out)) return true;
    return build_and_send_perfmgt(false, out);
}

bool IbMadClient::query_port_counters(IbPortCounters& out) {
    if (!ready_) return false;
#if defined(__linux__)
    if (use_mad_ && umad_fd_ >= 0 && query_via_mad(out)) return true;
#endif
    return query_via_sysfs(out);
}

bool IbMadClient::sample_util(double peak_bw_gbps, float& out_util) {
    const uint64_t now = steady_ns();
    // Batch MAD queries ≈ every 2 ms
    if (now - last_query_ns_ < 2'000'000ull && prev_ts_ns_ != 0) {
        out_util = cached_util_;
        return true;
    }
    last_query_ns_ = now;

    IbPortCounters c;
    if (!query_port_counters(c)) return false;

    // Extended counters are in bytes; base PortCounters are 4-byte words
    const uint64_t xmit_bytes =
        c.extended ? c.port_xmit_data : c.port_xmit_data * 4ull;
    const uint64_t rcv_bytes =
        c.extended ? c.port_rcv_data : c.port_rcv_data * 4ull;

    if (prev_ts_ns_ == 0) {
        prev_xmit_ = xmit_bytes;
        prev_rcv_ = rcv_bytes;
        prev_wait_ = c.xmit_wait;
        prev_ts_ns_ = now;
        cached_util_ = 0.0f;
        out_util = 0.0f;
        return true;
    }

    const double dt = (now - prev_ts_ns_) * 1e-9;
    const double dtx = static_cast<double>(xmit_bytes - prev_xmit_);
    const double drx = static_cast<double>(rcv_bytes - prev_rcv_);
    const double gbps = (dtx + drx) / (dt * 1e9 + 1e-12);
    double util = gbps / std::max(peak_bw_gbps, 1.0);

    // Fold xmit_wait growth as congestion signal (normalized softly)
    if (c.xmit_wait > prev_wait_) {
        const double dwait = static_cast<double>(c.xmit_wait - prev_wait_);
        util = std::max(util, std::min(0.99, util + dwait * 1e-6));
    }

    prev_xmit_ = xmit_bytes;
    prev_rcv_ = rcv_bytes;
    prev_wait_ = c.xmit_wait;
    prev_ts_ns_ = now;
    cached_util_ = static_cast<float>(std::clamp(util, 0.0, 0.99));
    out_util = cached_util_;
    return true;
}

// ---------------------------------------------------------------------------
// HwCounterBackend wrapping per-edge IbMadClient instances
// ---------------------------------------------------------------------------
class IbMadBackend final : public HwCounterBackend {
public:
    const char* name() const override { return name_.c_str(); }

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
        clients_.clear();
        clients_.resize(bindings.size());
        bool any = false;
        std::unordered_map<std::string, std::shared_ptr<IbMadClient>> shared;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (bindings[i].fabric != FabricType::InfiniBand) continue;
            const std::string key = bindings[i].ib_device + ":" +
                                    std::to_string(bindings[i].ib_port);
            auto it = shared.find(key);
            if (it == shared.end()) {
                auto c = std::make_shared<IbMadClient>();
                if (!c->open(bindings[i].ib_device, bindings[i].ib_port)) continue;
                name_ = c->backend_name();
                shared[key] = c;
                clients_[i] = c;
                any = true;
            } else {
                clients_[i] = it->second;
                any = true;
            }
        }
        return any;
    }

    bool sample(uint32_t edge_idx, float& out_util) override {
        if (edge_idx >= clients_.size() || !clients_[edge_idx]) return false;
        return clients_[edge_idx]->sample_util(bindings_[edge_idx].peak_bw_gbps,
                                               out_util);
    }

private:
    std::vector<EdgeHwBinding> bindings_;
    std::vector<std::shared_ptr<IbMadClient>> clients_;
    std::string name_{"ib_mad"};
};

std::unique_ptr<HwCounterBackend> make_ib_mad_backend() {
    return std::make_unique<IbMadBackend>();
}

}  // namespace hics
