#include "shim/ucx_transport.hpp"
#include "shim/shim.hpp"

namespace hics {

UcxTransport::UcxTransport(HICSShim& shim) : shim_(shim) {}

bool UcxTransport::open_iface(const UcxIfaceConfig& cfg) {
    cfg_ = cfg;
    open_ = true;
    return true;
}

void UcxTransport::close_iface() { open_ = false; }

bool UcxTransport::create_ep(const EndpointId& local, const EndpointId& remote,
                             UcxEndpoint& out) {
    if (!open_) return false;
    out.local = local;
    out.remote = remote;
    out.uct_ep = reinterpret_cast<void*>(
        (static_cast<uint64_t>(local.node_id) << 32) |
        (static_cast<uint64_t>(local.local_id) << 16) |
        (static_cast<uint64_t>(remote.node_id) << 8) | remote.local_id);
    return true;
}

int UcxTransport::zcopy_put(UcxEndpoint& ep, const void* buf, size_t len,
                            UcxRequest& req, TrafficClass cls, double deadline_us,
                            void* remote) {
    if (!open_) return -1;
    if (len > cfg_.max_zcopy_bytes) return -1;
    const double dl = deadline_us > 0 ? deadline_us : cfg_.default_deadline_us;
    req.transfer_id = shim_.submit_transfer(
        cls, ep.local, ep.remote, len, dl,
        const_cast<void*>(buf), remote);
    req.length = len;
    req.completed = false;
    return req.transfer_id == 0 ? -1 : 0;
}

int UcxTransport::zcopy_get(UcxEndpoint& ep, void* buf, size_t len, UcxRequest& req,
                            TrafficClass cls, double deadline_us,
                            const void* remote) {
    // Get: pull from remote into local buf (reverse put with swapped buffers)
    UcxEndpoint rev{ep.remote, ep.local, ep.uct_ep};
    return zcopy_put(rev, remote, len, req, cls, deadline_us, buf);
}

unsigned UcxTransport::progress(double dt_ms) {
    return static_cast<unsigned>(shim_.poll_transfers(dt_ms));
}

bool UcxTransport::request_completed(const UcxRequest& req) const {
    auto st = shim_.transfer_status(req.transfer_id);
    return st.state == TransferState::Completed || st.bytes_remaining == 0;
}

}  // namespace hics
