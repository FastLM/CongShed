#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hics {

enum class FabricType : uint8_t {
    NVLink = 0,
    PCIe = 1,
    CXL = 2,
    InfiniBand = 3,
};

enum class TrafficClass : uint8_t {
    TensorParallel = 0,  // TP all-reduce
    PipelineParallel = 1,  // PP activation send
    KVMigration = 2,       // KV cache migration
};

enum class EndpointType : uint8_t {
    GPU = 0,
    CPU = 1,
    CXLController = 2,
    IBPort = 3,
};

struct EndpointId {
    uint32_t node_id;
    uint32_t local_id;
    EndpointType type;

    bool operator==(const EndpointId& o) const {
        return node_id == o.node_id && local_id == o.local_id && type == o.type;
    }
};

struct EndpointIdHash {
    size_t operator()(const EndpointId& id) const {
        return (static_cast<size_t>(id.node_id) << 16) |
               (static_cast<size_t>(id.local_id) << 8) |
               static_cast<size_t>(id.type);
    }
};

struct EdgeAttributes {
    FabricType fabric;
    double bandwidth_gbps;   // peak bandwidth (GB/s)
    double latency_us;       // baseline latency (µs)
    double contention_alpha;
    int rail_id{-1};         // 0/1 for dual IB rails; -1 = N/A
};

// Default KV striping / suspension quantum
inline constexpr uint64_t kKvChunkBytes = 256ull * 1024ull * 1024ull;
inline constexpr int kIbRailCount = 2;

struct TransferRequest {
    EndpointId source;
    EndpointId dest;
    uint64_t size_bytes;
    TrafficClass traffic_class;
    double deadline_us;  // latency budget for this transfer
};

struct PathCost {
    std::vector<uint32_t> edge_indices;
    double latency_us;
};

// Fabric-specific contention exponents
inline double contention_exponent(FabricType fabric) {
    switch (fabric) {
        case FabricType::NVLink: return 1.4;
        case FabricType::InfiniBand: return 1.8;
        case FabricType::PCIe: return 2.1;
        case FabricType::CXL: return 2.1;
    }
    return 1.8;
}

// Default fabric bandwidth/latency; override per deployment as needed.
inline EdgeAttributes default_fabric_attrs(FabricType fabric, int rail_id = -1) {
    switch (fabric) {
        case FabricType::NVLink:
            return {FabricType::NVLink, 900.0, 1.0, 1.4, -1};
        case FabricType::PCIe:
            return {FabricType::PCIe, 64.0, 3.0, 2.1, -1};
        case FabricType::CXL:
            return {FabricType::CXL, 50.0, 5.0, 2.1, -1};
        case FabricType::InfiniBand:
            return {FabricType::InfiniBand, 50.0, 5.0, 1.8, rail_id};
    }
    return {FabricType::InfiniBand, 50.0, 5.0, 1.8, rail_id};
}

}  // namespace hics
