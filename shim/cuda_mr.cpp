#include "shim/cuda_mr.hpp"

#include <cstring>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace hics {
namespace {

// Mirror of cudaPointerAttributes (stable fields we need)
struct CudaPointerAttributes {
    int type;  // 0 unregistered, 1 host, 2 device, 3 managed (CUDA 11+)
    int device;
    void* devicePointer;
    void* hostPointer;
};

constexpr int kCudaSuccess = 0;
constexpr int kCudaMemcpyDeviceToHost = 2;
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr unsigned kCudaHostRegisterDefault = 0;
constexpr unsigned kCudaHostRegisterMapped = 0x02;

}  // namespace

CudaRuntime& CudaRuntime::instance() {
    static CudaRuntime rt;
    return rt;
}

CudaRuntime::CudaRuntime() {
#if defined(__linux__) || defined(__APPLE__)
    const char* libs[] = {
        "libcudart.so.12", "libcudart.so.11.0", "libcudart.so.11",
        "libcudart.so", "libcudart.dylib", nullptr};
    for (int i = 0; libs[i]; ++i) {
        handle_ = dlopen(libs[i], RTLD_LAZY | RTLD_LOCAL);
        if (handle_) break;
    }
    if (!handle_) return;

    cudaGetDeviceCount_ =
        reinterpret_cast<int (*)(int*)>(dlsym(handle_, "cudaGetDeviceCount"));
    cudaPointerGetAttributes_ = reinterpret_cast<int (*)(void*, const void*)>(
        dlsym(handle_, "cudaPointerGetAttributes"));
    cudaHostRegister_ = reinterpret_cast<int (*)(void*, size_t, unsigned)>(
        dlsym(handle_, "cudaHostRegister"));
    cudaHostUnregister_ =
        reinterpret_cast<int (*)(void*)>(dlsym(handle_, "cudaHostUnregister"));
    cudaHostGetDevicePointer_ =
        reinterpret_cast<int (*)(void**, void*, unsigned)>(
            dlsym(handle_, "cudaHostGetDevicePointer"));
    cudaStreamSynchronize_ =
        reinterpret_cast<int (*)(void*)>(dlsym(handle_, "cudaStreamSynchronize"));
    cudaMemcpyAsync_ =
        reinterpret_cast<int (*)(void*, const void*, size_t, int, void*)>(
            dlsym(handle_, "cudaMemcpyAsync"));
    cudaGetLastError_ = reinterpret_cast<int (*)()>(dlsym(handle_, "cudaGetLastError"));

    if (!cudaGetDeviceCount_ || !cudaPointerGetAttributes_) {
        dlclose(handle_);
        handle_ = nullptr;
        return;
    }
    int n = 0;
    if (cudaGetDeviceCount_(&n) != kCudaSuccess || n <= 0) {
        // Driver present but no GPU — still allow host registration paths
        device_count_ = 0;
    } else {
        device_count_ = n;
    }
    ready_ = true;
#endif
}

CudaRuntime::~CudaRuntime() {
#if defined(__linux__) || defined(__APPLE__)
    if (handle_) dlclose(handle_);
#endif
}

bool CudaRuntime::is_device_pointer(const void* ptr, int* device_id) const {
    if (!ready_ || !cudaPointerGetAttributes_ || !ptr) return false;
    CudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes_(&attrs, ptr) != kCudaSuccess) return false;
    // type: 2 = device (cudaMemoryTypeDevice)
    if (attrs.type == 2 || attrs.devicePointer == ptr) {
        if (device_id) *device_id = attrs.device;
        return true;
    }
    return false;
}

bool CudaRuntime::is_host_registered(const void* ptr) const {
    if (!ready_ || !cudaPointerGetAttributes_ || !ptr) return false;
    CudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes_(&attrs, ptr) != kCudaSuccess) return false;
    return attrs.type == 1 || attrs.hostPointer != nullptr;
}

bool CudaRuntime::host_register(void* ptr, size_t size) {
    if (!ptr || size == 0) return false;
    if (ready_ && cudaHostRegister_) {
        // Mapped so device can DMA from pinned host pages
        int st = cudaHostRegister_(ptr, size, kCudaHostRegisterMapped);
        if (st == kCudaSuccess) return true;
        st = cudaHostRegister_(ptr, size, kCudaHostRegisterDefault);
        if (st == kCudaSuccess) return true;
    }
#if defined(__linux__)
    // Fallback: pin via mlock when CUDA absent
    return mlock(ptr, size) == 0;
#else
    return false;
#endif
}

bool CudaRuntime::host_unregister(void* ptr) {
    if (!ptr) return false;
    if (ready_ && cudaHostUnregister_) {
        return cudaHostUnregister_(ptr) == kCudaSuccess;
    }
#if defined(__linux__)
    // munlock needs size — best-effort skip when CUDA path unused
    (void)ptr;
    return true;
#else
    return true;
#endif
}

bool CudaRuntime::host_get_device_pointer(void** device_ptr, void* host_ptr) {
    if (!device_ptr || !host_ptr) return false;
    if (ready_ && cudaHostGetDevicePointer_) {
        if (cudaHostGetDevicePointer_(device_ptr, host_ptr, 0) == kCudaSuccess)
            return true;
    }
    *device_ptr = host_ptr;
    return true;
}

bool CudaRuntime::stream_sync_default() {
    if (!ready_ || !cudaStreamSynchronize_) return true;
    return cudaStreamSynchronize_(nullptr) == kCudaSuccess;
}

bool CudaRuntime::memcpy_dtoh_async(void* host, const void* device, size_t size) {
    if (!ready_ || !cudaMemcpyAsync_) return false;
    return cudaMemcpyAsync_(host, device, size, kCudaMemcpyDeviceToHost, nullptr) ==
           kCudaSuccess;
}

bool CudaRuntime::memcpy_htod_async(void* device, const void* host, size_t size) {
    if (!ready_ || !cudaMemcpyAsync_) return false;
    return cudaMemcpyAsync_(device, host, size, kCudaMemcpyHostToDevice, nullptr) ==
           kCudaSuccess;
}

MemoryRegistry& MemoryRegistry::instance() {
    static MemoryRegistry reg;
    return reg;
}

int MemoryRegistry::supported_ptr_types() const {
    int mask = static_cast<int>(MrPtrType::Host);
    if (CudaRuntime::instance().available() &&
        CudaRuntime::instance().device_count() > 0) {
        mask |= static_cast<int>(MrPtrType::Cuda);
    }
    return mask;
}

void* MemoryRegistry::register_mr(void* data, size_t size, int nccl_ptr_type) {
    if (!data || size == 0) return nullptr;
    auto& cuda = CudaRuntime::instance();

    MemoryRegion mr;
    mr.addr = data;
    mr.size = size;
    mr.type = (nccl_ptr_type & static_cast<int>(MrPtrType::Cuda)) ? MrPtrType::Cuda
                                                                  : MrPtrType::Host;

    if (mr.type == MrPtrType::Cuda || cuda.is_device_pointer(data, &mr.device_id)) {
        mr.type = MrPtrType::Cuda;
        mr.device_ptr = data;
        mr.cuda_registered = true;
        // Device memory is already GPU-resident; no host pin needed.
    } else {
        mr.type = MrPtrType::Host;
        if (cuda.host_register(data, size)) {
            mr.host_registered = true;
            void* dptr = nullptr;
            if (cuda.host_get_device_pointer(&dptr, data)) mr.device_ptr = dptr;
        } else {
            mr.device_ptr = data;
        }
    }

    std::lock_guard<std::mutex> lock(mu_);
    mr.id = next_id_++;
    auto id = mr.id;
    regions_[id] = mr;
    return reinterpret_cast<void*>(id);
}

bool MemoryRegistry::deregister_mr(void* mhandle) {
    if (!mhandle) return false;
    const uint64_t id = reinterpret_cast<uint64_t>(mhandle);
    std::lock_guard<std::mutex> lock(mu_);
    auto it = regions_.find(id);
    if (it == regions_.end()) return false;
    auto& mr = it->second;
    if (mr.refcount > 1) {
        --mr.refcount;
        return true;
    }
    if (mr.host_registered) {
        CudaRuntime::instance().host_unregister(mr.addr);
    }
    regions_.erase(it);
    return true;
}

bool MemoryRegistry::get_mr(void* mhandle, MemoryRegion& out) const {
    if (!mhandle) return false;
    const uint64_t id = reinterpret_cast<uint64_t>(mhandle);
    std::lock_guard<std::mutex> lock(mu_);
    auto it = regions_.find(id);
    if (it == regions_.end()) return false;
    out = it->second;
    return true;
}

bool MemoryRegistry::flush(void* mhandle) {
    MrPtrType type = MrPtrType::Host;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (mhandle) {
            const uint64_t id = reinterpret_cast<uint64_t>(mhandle);
            auto it = regions_.find(id);
            if (it != regions_.end()) {
                type = it->second.type;
                found = true;
            }
        }
    }
    (void)type;
    (void)found;
    return CudaRuntime::instance().stream_sync_default();
}

size_t MemoryRegistry::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return regions_.size();
}

}  // namespace hics
