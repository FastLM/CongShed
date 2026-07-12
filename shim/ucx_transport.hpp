#pragma once

#include "model/model.hpp"
#include "transfer/transfer_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace hics {

class HICSShim;

// UCX-style custom transport component for KV migrations / P2P.
// Mirrors the uct_iface surface HICS registers as a preferred transport.
struct UcxIfaceConfig {
    std::string device{"hics0"};
    size_t max_zcopy_bytes{1ull << 30};
    double default_deadline_us{50000.0};
};

struct UcxEndpoint {
    EndpointId local;
    EndpointId remote;
    void* uct_ep{nullptr};
};

struct UcxRequest {
    TransferId transfer_id{0};
    size_t length{0};
    bool completed{false};
};

class UcxTransport {
public:
    explicit UcxTransport(HICSShim& shim);

    bool open_iface(const UcxIfaceConfig& cfg = {});
    void close_iface();

    // uct_ep_create analogue
    bool create_ep(const EndpointId& local, const EndpointId& remote, UcxEndpoint& out);

    // zcopy put / get used for KV cache chunks
    int zcopy_put(UcxEndpoint& ep, const void* buf, size_t len, UcxRequest& req,
                  TrafficClass cls = TrafficClass::KVMigration,
                  double deadline_us = -1.0);
    int zcopy_get(UcxEndpoint& ep, void* buf, size_t len, UcxRequest& req,
                  TrafficClass cls = TrafficClass::KVMigration,
                  double deadline_us = -1.0);

    // Progress engine — drains transfer executor
    unsigned progress(double dt_ms = 0.2);

    bool request_completed(const UcxRequest& req) const;

    const UcxIfaceConfig& config() const { return cfg_; }
    bool iface_open() const { return open_; }

private:
    HICSShim& shim_;
    UcxIfaceConfig cfg_{};
    bool open_{false};
};

}  // namespace hics
