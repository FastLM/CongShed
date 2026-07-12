#pragma once

#include <cstddef>
#include <cstdint>

// NCCL Net plugin ABI (ncclNet_v8-compatible surface).

#ifdef HICS_HAVE_NCCL_NET_H
#include <net.h>
#else

enum ncclResult_t {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7
};

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

typedef struct {
    char name[64];
    union {
        void* ptr;
        int fd;
    };
} ncclNetHandle_t;

typedef enum {
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags,
                                  const char* file, int line, const char* fmt, ...);

struct ncclNetProperties_v8 {
    char* name;
    char* pciPath;
    uint64_t guid;
    int ptrSupport;
    int regIsGlobal;
    int speed;
    int port;
    float latency;
    int maxComms;
    int maxRecvs;
    int netDeviceType;
    int netDeviceVersion;
};

struct ncclNet_v8 {
    const char* name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8* props);
    ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
    ncclResult_t (*connect)(int dev, void* handle, void** sendComm,
                            void** sendDevComm);
    ncclResult_t (*accept)(void* listenComm, void** recvComm, void** recvDevComm);
    ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type,
                          void** mhandle);
    ncclResult_t (*deregMr)(void* comm, void* mhandle);
    ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                          void* mhandle, void** request);
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes,
                          int* tags, void** mhandles, void** request);
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                           void** mhandles, void** request);
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
};

#endif  // HICS_HAVE_NCCL_NET_H

namespace hics {

bool nccl_plugin_bootstrap(const char* weights_path = nullptr);

extern "C" {
extern ncclNet_v8 ncclNetPlugin_v8;

int hics_plugin_init();
int hics_plugin_devices(int* ndev);
int hics_plugin_isend(void* send_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request);
int hics_plugin_irecv(void* recv_comm, void* data, size_t size, int tag,
                      void* mhandle, void** request);
int hics_plugin_test(void* request, int* done, int* size);
}

}  // namespace hics
