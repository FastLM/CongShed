#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hics {

enum class MrPtrType : int {
    Host = 0x1,  // NCCL_PTR_HOST
    Cuda = 0x2,  // NCCL_PTR_CUDA
};

struct MemoryRegion {
    uint64_t id{0};
    void* addr{nullptr};
    size_t size{0};
    MrPtrType type{MrPtrType::Host};
    void* device_ptr{nullptr};   // CUDA device pointer (may equal addr for device mem)
    int device_id{-1};
    bool host_registered{false}; // cudaHostRegister / mlock
    bool cuda_registered{false};
    uint32_t refcount{1};
};

// CUDA Runtime (dlopen) helpers used by NCCL regMr / iflush.
class CudaRuntime {
public:
    static CudaRuntime& instance();

    bool available() const { return ready_; }
    int device_count() const { return device_count_; }

    // Pointer classification
    bool is_device_pointer(const void* ptr, int* device_id = nullptr) const;
    bool is_host_registered(const void* ptr) const;

    // Registration
    bool host_register(void* ptr, size_t size);
    bool host_unregister(void* ptr);
    bool host_get_device_pointer(void** device_ptr, void* host_ptr);

    // Stream sync used by iflush for CUDA buffers
    bool stream_sync_default();
    bool memcpy_dtoh_async(void* host, const void* device, size_t size);
    bool memcpy_htod_async(void* device, const void* host, size_t size);

private:
    CudaRuntime();
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    bool ready_{false};
    int device_count_{0};
#if defined(__linux__) || defined(__APPLE__)
    void* handle_{nullptr};
    int (*cudaGetDeviceCount_)(int*){nullptr};
    int (*cudaPointerGetAttributes_)(void*, const void*){nullptr};
    int (*cudaHostRegister_)(void*, size_t, unsigned){nullptr};
    int (*cudaHostUnregister_)(void*){nullptr};
    int (*cudaHostGetDevicePointer_)(void**, void*, unsigned){nullptr};
    int (*cudaStreamSynchronize_)(void*){nullptr};
    int (*cudaMemcpyAsync_)(void*, const void*, size_t, int, void*){nullptr};
    int (*cudaGetLastError_)(){nullptr};
#endif
};

// Process-wide MR table for NCCL plugin regMr / deregMr.
class MemoryRegistry {
public:
    static MemoryRegistry& instance();

    // Register a buffer; returns opaque mhandle (non-null on success).
    void* register_mr(void* data, size_t size, int nccl_ptr_type);
    bool deregister_mr(void* mhandle);
    bool get_mr(void* mhandle, MemoryRegion& out) const;

    // Ensure CUDA visibility before NCCL test completion (host↔device).
    bool flush(void* mhandle);

    size_t size() const;
    int supported_ptr_types() const;  // bitmask HOST | CUDA

private:
    MemoryRegistry() = default;
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, MemoryRegion> regions_;
    uint64_t next_id_{1};
};

}  // namespace hics
