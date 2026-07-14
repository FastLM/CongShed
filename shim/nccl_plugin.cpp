#include "shim/nccl_plugin.hpp"
#include "shim/cuda_mr.hpp"
#include "shim/shim.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hics {
namespace {

struct PluginComm {
    int peer_rank{0};
    int local_gpu{0};
    bool is_send{true};
};

struct PluginRequest {
    TransferId transfer_id{0};
    size_t size{0};
    bool done{false};
    int result_size{0};
    void* mhandle{nullptr};
    void* data{nullptr};
    bool is_cuda{false};
};

std::mutex g_mu;
std::unique_ptr<HICSShim> g_shim;
std::atomic<uint64_t> g_req_ids{1};
std::unordered_map<uint64_t, PluginRequest> g_requests;
ncclDebugLogger_t g_logger = nullptr;
std::string g_weights_path;

EndpointId gpu_ep(int rank) {
    return EndpointId{static_cast<uint32_t>(rank / 8),
                      static_cast<uint32_t>(rank % 8), EndpointType::GPU};
}

TrafficClass classify_tag(int tag, size_t size) {
    if (size >= (256ull * 1024ull * 1024ull)) return TrafficClass::KVMigration;
    if (tag >= 1000) return TrafficClass::KVMigration;
    if (tag >= 100) return TrafficClass::PipelineParallel;
    return TrafficClass::TensorParallel;
}

void log_info(const char* msg) {
    if (g_logger) {
        g_logger(NCCL_LOG_INFO, 0, __FILE__, __LINE__, "%s", msg);
    }
}

}  // namespace

bool nccl_plugin_bootstrap(const char* weights_path) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_shim) return true;
    if (weights_path) g_weights_path = weights_path;
    g_shim = std::make_unique<HICSShim>();
    HICSShim::InitOptions opts;
    opts.weights_path = g_weights_path;
    opts.enable_executor = true;
    return g_shim->initialize(opts);
}

namespace {

ncclResult_t plugin_init(ncclDebugLogger_t logFunction) {
    g_logger = logFunction;
    if (!nccl_plugin_bootstrap(nullptr)) return ncclInternalError;
    // Touch CUDA runtime early so ptrSupport reflects reality
    (void)CudaRuntime::instance().available();
    log_info("HICS NCCL net plugin initialized");
    return ncclSuccess;
}

ncclResult_t plugin_devices(int* ndev) {
    if (!ndev) return ncclInvalidArgument;
    *ndev = 1;
    return ncclSuccess;
}

ncclResult_t plugin_getProperties(int /*dev*/, ncclNetProperties_v8* props) {
    if (!props) return ncclInvalidArgument;
    static char name[] = "hics";
    static char pci[] = "";
    props->name = name;
    props->pciPath = pci;
    props->guid = 0x48494353ull;
    props->ptrSupport = MemoryRegistry::instance().supported_ptr_types();
    props->regIsGlobal = 0;
    props->speed = 400000;
    props->port = 0;
    props->latency = 0.8f;
    props->maxComms = 65536;
    props->maxRecvs = 64;
    props->netDeviceType = 0;
    props->netDeviceVersion = 0;
    return ncclSuccess;
}

ncclResult_t plugin_listen(int /*dev*/, void* handle, void** listenComm) {
    if (!listenComm) return ncclInvalidArgument;
    auto* comm = new PluginComm{};
    comm->is_send = false;
    if (handle) {
        auto* h = static_cast<ncclNetHandle_t*>(handle);
        std::snprintf(h->name, sizeof(h->name), "hics");
        h->ptr = comm;
    }
    *listenComm = comm;
    return ncclSuccess;
}

ncclResult_t plugin_connect(int /*dev*/, void* handle, void** sendComm,
                            void** sendDevComm) {
    if (!sendComm) return ncclInvalidArgument;
    auto* comm = new PluginComm{};
    comm->is_send = true;
    if (handle) {
        auto* h = static_cast<ncclNetHandle_t*>(handle);
        if (h->ptr) {
            auto* listen = static_cast<PluginComm*>(h->ptr);
            comm->peer_rank = listen->local_gpu;
        }
    }
    *sendComm = comm;
    if (sendDevComm) *sendDevComm = nullptr;
    return ncclSuccess;
}

ncclResult_t plugin_accept(void* listenComm, void** recvComm, void** recvDevComm) {
    if (!listenComm || !recvComm) return ncclInvalidArgument;
    auto* listen = static_cast<PluginComm*>(listenComm);
    auto* comm = new PluginComm{};
    *comm = *listen;
    *recvComm = comm;
    if (recvDevComm) *recvDevComm = nullptr;
    return ncclSuccess;
}

ncclResult_t plugin_regMr(void* /*comm*/, void* data, size_t size, int type,
                          void** mhandle) {
    if (!mhandle || !data || size == 0) return ncclInvalidArgument;
    void* handle = MemoryRegistry::instance().register_mr(data, size, type);
    if (!handle) return ncclSystemError;
    *mhandle = handle;
    return ncclSuccess;
}

ncclResult_t plugin_deregMr(void* /*comm*/, void* mhandle) {
    if (!mhandle) return ncclSuccess;
    return MemoryRegistry::instance().deregister_mr(mhandle) ? ncclSuccess
                                                             : ncclInvalidArgument;
}

ncclResult_t plugin_isend(void* sendComm, void* data, size_t size, int tag,
                          void* mhandle, void** request) {
    if (!sendComm || !request) return ncclInvalidArgument;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_shim) return ncclInternalError;

    // Validate MR when provided; allow unregistered host for small msgs
    bool is_cuda = false;
    if (mhandle) {
        MemoryRegion mr;
        if (!MemoryRegistry::instance().get_mr(mhandle, mr)) return ncclInvalidUsage;
        is_cuda = (mr.type == MrPtrType::Cuda);
        if (mr.addr != data && data != nullptr) {
            auto* base = static_cast<uint8_t*>(mr.addr);
            auto* ptr = static_cast<uint8_t*>(data);
            if (ptr < base || ptr + size > base + mr.size) return ncclInvalidArgument;
        }
    } else if (data) {
        int dev = -1;
        is_cuda = CudaRuntime::instance().is_device_pointer(data, &dev);
    }

    auto* comm = static_cast<PluginComm*>(sendComm);
    const TrafficClass cls = classify_tag(tag, size);
    const double deadline =
        (cls == TrafficClass::TensorParallel) ? 50.0 : 5e4;

    TransferId tid = g_shim->submit_transfer(
        cls, gpu_ep(comm->local_gpu), gpu_ep(comm->peer_rank), size, deadline);

    const uint64_t rid = g_req_ids.fetch_add(1);
    g_requests[rid] =
        PluginRequest{tid, size, false, static_cast<int>(size), mhandle, data, is_cuda};
    *request = reinterpret_cast<void*>(rid);
    return ncclSuccess;
}

ncclResult_t plugin_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                          void** mhandles, void** request) {
    if (!recvComm || !request || n < 1) return ncclInvalidArgument;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_shim) return ncclInternalError;

    auto* comm = static_cast<PluginComm*>(recvComm);
    size_t size = sizes ? sizes[0] : 0;
    int tag = tags ? tags[0] : 0;
    void* mhandle = (mhandles && mhandles[0]) ? mhandles[0] : nullptr;
    void* buf = (data && data[0]) ? data[0] : nullptr;

    bool is_cuda = false;
    if (mhandle) {
        MemoryRegion mr;
        if (!MemoryRegistry::instance().get_mr(mhandle, mr)) return ncclInvalidUsage;
        is_cuda = (mr.type == MrPtrType::Cuda);
    } else if (buf) {
        int dev = -1;
        is_cuda = CudaRuntime::instance().is_device_pointer(buf, &dev);
    }

    const TrafficClass cls = classify_tag(tag, size);
    TransferId tid = g_shim->submit_transfer(
        cls, gpu_ep(comm->peer_rank), gpu_ep(comm->local_gpu), size,
        cls == TrafficClass::TensorParallel ? 50.0 : 5e4);

    const uint64_t rid = g_req_ids.fetch_add(1);
    g_requests[rid] =
        PluginRequest{tid, size, false, static_cast<int>(size), mhandle, buf, is_cuda};
    *request = reinterpret_cast<void*>(rid);
    return ncclSuccess;
}

ncclResult_t plugin_iflush(void* /*recvComm*/, int n, void** /*data*/, int* /*sizes*/,
                           void** mhandles, void** request) {
    // Ensure CUDA writes are visible to host / peer before NCCL completes recv
    for (int i = 0; i < n; ++i) {
        void* mh = (mhandles && mhandles[i]) ? mhandles[i] : nullptr;
        MemoryRegistry::instance().flush(mh);
    }
    if (request) *request = reinterpret_cast<void*>(0);
    return ncclSuccess;
}

ncclResult_t plugin_test(void* request, int* done, int* sizes) {
    if (!done) return ncclInvalidArgument;
    if (!request) {
        *done = 1;
        return ncclSuccess;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    const uint64_t rid = reinterpret_cast<uint64_t>(request);
    auto it = g_requests.find(rid);
    if (it == g_requests.end()) {
        *done = 1;
        return ncclSuccess;
    }
    if (g_shim) g_shim->poll_transfers(0.2);
    auto prog = g_shim->transfer_status(it->second.transfer_id);
    if (prog.state == TransferState::Completed ||
        prog.state == TransferState::Cancelled ||
        prog.bytes_remaining == 0) {
        // CUDA path: sync before reporting done so consumer sees data
        if (it->second.is_cuda || it->second.mhandle) {
            MemoryRegistry::instance().flush(it->second.mhandle);
        }
        it->second.done = true;
        *done = 1;
        if (sizes) sizes[0] = it->second.result_size;
        g_requests.erase(it);
    } else {
        *done = 0;
    }
    return ncclSuccess;
}

ncclResult_t plugin_closeSend(void* sendComm) {
    delete static_cast<PluginComm*>(sendComm);
    return ncclSuccess;
}
ncclResult_t plugin_closeRecv(void* recvComm) {
    delete static_cast<PluginComm*>(recvComm);
    return ncclSuccess;
}
ncclResult_t plugin_closeListen(void* listenComm) {
    delete static_cast<PluginComm*>(listenComm);
    return ncclSuccess;
}

}  // namespace

extern "C" {

ncclNet_v8 ncclNetPlugin_v8 = {
    "hics",
    plugin_init,
    plugin_devices,
    plugin_getProperties,
    plugin_listen,
    plugin_connect,
    plugin_accept,
    plugin_regMr,
    plugin_deregMr,
    plugin_isend,
    plugin_irecv,
    plugin_iflush,
    plugin_test,
    plugin_closeSend,
    plugin_closeRecv,
    plugin_closeListen,
};

int hics_plugin_init() { return nccl_plugin_bootstrap(nullptr) ? 0 : -1; }

int hics_plugin_devices(int* ndev) {
    return plugin_devices(ndev) == ncclSuccess ? 0 : -1;
}

int hics_plugin_isend(void* send_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request) {
    return plugin_isend(send_comm, data, size, tag, mhandle, request) == ncclSuccess
               ? 0
               : -1;
}

int hics_plugin_irecv(void* recv_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request) {
    size_t sizes[1] = {size};
    int tags[1] = {tag};
    void* datas[1] = {data};
    void* mhs[1] = {mhandle};
    return plugin_irecv(recv_comm, 1, datas, sizes, tags, mhs, request) == ncclSuccess
               ? 0
               : -1;
}

int hics_plugin_test(void* request, int* done, int* size) {
    return plugin_test(request, done, size) == ncclSuccess ? 0 : -1;
}

}  // extern "C"

}  // namespace hics
