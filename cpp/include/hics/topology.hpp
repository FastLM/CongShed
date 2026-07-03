#pragma once

#include "hics/types.hpp"

#include <unordered_map>
#include <vector>

namespace hics {

struct GraphEdge {
    uint32_t src_vertex;
    uint32_t dst_vertex;
    EdgeAttributes attrs;
};

// Latency model from Eq. (1): λ_ij(s,t) = ℓ_ij + s / (B_ij * (1 - u_ij(t))^α)
double edge_latency_us(const EdgeAttributes& attrs, uint64_t size_bytes,
                       double utilization);

class TopologyGraph {
public:
    uint32_t add_vertex(EndpointId endpoint);
    uint32_t add_edge(uint32_t src, uint32_t dst, EdgeAttributes attrs);

    const std::vector<EndpointId>& vertices() const { return vertices_; }
    const std::vector<GraphEdge>& edges() const { return edges_; }

    // Pre-computed k-shortest paths (k=6) per (src, dst) pair (§VI)
    void precompute_paths(uint32_t k = 6, uint32_t max_hops = 5);
    const std::vector<std::vector<uint32_t>>& paths(uint32_t src, uint32_t dst) const;

    uint32_t vertex_of(const EndpointId& ep) const;

    // Build default 8-GPU node topology (Fig. 2)
    static TopologyGraph build_h100_node(uint32_t node_id, uint32_t num_gpus = 8);

    // Build 64-GPU cluster (8 nodes × 8 GPUs)
    static TopologyGraph build_cluster(uint32_t num_nodes = 8, uint32_t gpus_per_node = 8);

private:
    std::vector<EndpointId> vertices_;
    std::vector<GraphEdge> edges_;
    std::unordered_map<uint64_t, std::vector<std::vector<uint32_t>>> path_cache_;

    static uint64_t path_key(uint32_t src, uint32_t dst) {
        return (static_cast<uint64_t>(src) << 32) | dst;
    }

    std::vector<std::vector<uint32_t>> yen_k_shortest(
        uint32_t src, uint32_t dst, uint32_t k, uint32_t max_hops) const;
};

}  // namespace hics
