#pragma once

#include "telemetry/hw_counters.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hics {

// Port performance counters from IBTA PerfMgt MAD (class 0x04).
struct IbPortCounters {
    uint64_t port_xmit_data{0};      // 4-byte words (base) or bytes (ext)
    uint64_t port_rcv_data{0};
    uint64_t port_xmit_pkts{0};
    uint64_t port_rcv_pkts{0};
    uint64_t symbol_err{0};
    uint64_t link_err{0};
    uint64_t xmit_wait{0};           // congestion indicator
    uint64_t port_xmit_discards{0};
    bool extended{false};            // PortCountersExtended (64-bit)
    bool valid{false};
};

// InfiniBand MAD client: prefers libibumad/libibmad (dlopen), falls back to
// sysfs counter files. Batches queries on a ~2 ms cadence.
class IbMadClient {
public:
    IbMadClient();
    ~IbMadClient();

    bool available() const { return ready_; }
    const char* backend_name() const { return backend_name_.c_str(); }

    // Open MAD port for device (e.g. "mlx5_0") / port number.
    bool open(const std::string& ca_name, int port);
    void close();

    // Issue PerfMgt GET — PortCountersExtended when supported, else PortCounters.
    bool query_port_counters(IbPortCounters& out);

    // Rate-limited refresh used by the telemetry backend (≤ every 2 ms).
    bool sample_util(double peak_bw_gbps, float& out_util);

private:
    bool ready_{false};
    bool use_mad_{false};
    std::string ca_name_;
    int port_{1};
    std::string backend_name_{"none"};
    std::string sysfs_base_;

    uint64_t prev_xmit_{0};
    uint64_t prev_rcv_{0};
    uint64_t prev_wait_{0};
    uint64_t prev_ts_ns_{0};
    uint64_t last_query_ns_{0};
    float cached_util_{0.0f};

#if defined(__linux__)
    void* umad_handle_{nullptr};
    void* mad_handle_{nullptr};
    int umad_fd_{-1};
    int umad_agent_{-1};
    int (*umad_init_)(){nullptr};
    int (*umad_done_)(){nullptr};
    int (*umad_open_port_)(const char*, int){nullptr};
    int (*umad_close_port_)(int){nullptr};
    int (*umad_register_)(int, int, int, uint32_t, void*){nullptr};
    int (*umad_unregister_)(int, int){nullptr};
    void* (*umad_alloc_)(int, size_t){nullptr};
    void (*umad_free_)(void*){nullptr};
    int (*umad_send_)(int, int, void*, int, int, int){nullptr};
    int (*umad_recv_)(int, void*, int*, int){nullptr};
    void* (*umad_get_mad_)(void*){nullptr};
    size_t (*umad_size_)(){nullptr};
#endif

    bool init_mad_libs();
    bool query_via_mad(IbPortCounters& out);
    bool query_via_sysfs(IbPortCounters& out);
    bool build_and_send_perfmgt(bool extended, IbPortCounters& out);
};

std::unique_ptr<HwCounterBackend> make_ib_mad_backend();

}  // namespace hics
