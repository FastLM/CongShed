#pragma once

#include "model/model.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace hics {

// Discovered IB HCA (one per rail when dual-NIC).
struct IbDeviceInfo {
    std::string name;   // e.g. mlx5_0
    int port{1};
    int rail_id{0};
    uint64_t guid{0};
};

// Opaque MR / QP handles for the dlopen ibverbs backend.
struct IbMrHandle {
    uint64_t id{0};
    void* addr{nullptr};
    size_t length{0};
    uint32_t lkey{0};
    uint32_t rkey{0};
    int rail_id{0};
    bool registered{false};
};

struct IbWorkRequest {
    uint64_t wr_id{0};
    int rail_id{0};
    void* local_addr{nullptr};
    void* remote_addr{nullptr};
    size_t length{0};
    bool completed{false};
    bool success{false};
};

// Dual-rail ibverbs transport. Prefer real libibverbs when present; otherwise
// perform host-side buffer copies so KV contents still move correctly.
class IbverbsTransport {
public:
    static IbverbsTransport& instance();

    // Discover HCAs under /sys/class/infiniband (or ibv_get_device_list).
    // Returns up to kIbRailCount devices, mapped to rail 0..N-1.
    bool init(int preferred_rails = kIbRailCount);
    void shutdown();

    bool available() const { return ready_; }
    bool hardware() const { return hardware_; }
    int rail_count() const { return static_cast<int>(devices_.size()); }
    const std::vector<IbDeviceInfo>& devices() const { return devices_; }

    // Register / deregister a buffer on a rail (ibv_reg_mr or host pin).
    bool reg_mr(int rail_id, void* addr, size_t length, IbMrHandle& out);
    bool dereg_mr(IbMrHandle& mr);

    // Post an RDMA WRITE (or host memcpy fallback) on `rail_id`.
    // For the loopback/demo path, remote_addr may equal a peer host buffer.
    int post_write(int rail_id, void* local, void* remote, size_t length,
                   const IbMrHandle* local_mr, IbWorkRequest& wr);

    // Poll completions; returns number newly completed.
    unsigned poll(int max_cqe = 16);

    // Synchronous helper used by TransferExecutor for chunk I/O.
    bool transfer_chunk(int rail_id, void* src, void* dst, size_t length);

    // Map a topology edge's rail_id attribute → device index.
    int device_for_rail(int rail_id) const;

private:
    IbverbsTransport() = default;
    ~IbverbsTransport();
    IbverbsTransport(const IbverbsTransport&) = delete;
    IbverbsTransport& operator=(const IbverbsTransport&) = delete;

    bool ready_{false};
    bool hardware_{false};
    std::vector<IbDeviceInfo> devices_;
    mutable std::mutex mu_;
    uint64_t next_wr_{1};
    uint64_t next_mr_{1};
    std::vector<IbWorkRequest> pending_;

#if defined(__linux__)
    void* ibv_handle_{nullptr};
    bool load_ibverbs();
    void unload_ibverbs();
#endif

    bool discover_sysfs(int preferred_rails);
    bool memcpy_transfer(void* src, void* dst, size_t length);
};

// Enumerate IB device names (sysfs), used by telemetry bindings too.
std::vector<std::string> discover_ib_device_names();

}  // namespace hics
