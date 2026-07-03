#include "hics/topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

namespace hics {

double edge_latency_us(const EdgeAttributes& attrs, uint64_t size_bytes,
                       double utilization) {
    utilization = std::clamp(utilization, 0.0, 0.99);
    double residual = 1.0 - utilization;
    double transfer_s = static_cast<double>(size_bytes) / 1e9 /
                        (attrs.bandwidth_gbps * std::pow(residual, attrs.contention_alpha));
    return attrs.latency_us + transfer_s * 1e6;
}

uint32_t TopologyGraph::add_vertex(EndpointId endpoint) {
    vertices_.push_back(endpoint);
    return static_cast<uint32_t>(vertices_.size() - 1);
}

uint32_t TopologyGraph::add_edge(uint32_t src, uint32_t dst, EdgeAttributes attrs) {
    edges_.push_back({src, dst, attrs});
    return static_cast<uint32_t>(edges_.size() - 1);
}

uint32_t TopologyGraph::vertex_of(const EndpointId& ep) const {
    for (size_t i = 0; i < vertices_.size(); ++i) {
        if (vertices_[i] == ep) return static_cast<uint32_t>(i);
    }
    return UINT32_MAX;
}

void TopologyGraph::precompute_paths(uint32_t k, uint32_t max_hops) {
    path_cache_.clear();
    const uint32_t n = static_cast<uint32_t>(vertices_.size());
    for (uint32_t src = 0; src < n; ++src) {
        for (uint32_t dst = 0; dst < n; ++dst) {
            if (src == dst) continue;
            path_cache_[path_key(src, dst)] = yen_k_shortest(src, dst, k, max_hops);
        }
    }
}

const std::vector<std::vector<uint32_t>>& TopologyGraph::paths(uint32_t src,
                                                               uint32_t dst) const {
    static const std::vector<std::vector<uint32_t>> empty;
    auto it = path_cache_.find(path_key(src, dst));
    return it != path_cache_.end() ? it->second : empty;
}

std::vector<std::vector<uint32_t>> TopologyGraph::yen_k_shortest(
    uint32_t src, uint32_t dst, uint32_t k, uint32_t max_hops) const {
    // Simplified Yen's k-shortest paths using BFS with hop limit
    struct State {
        uint32_t vertex;
        std::vector<uint32_t> edge_path;
        double cost;
        bool operator>(const State& o) const { return cost > o.cost; }
    };

    std::vector<std::vector<uint32_t>> result;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.push({src, {}, 0.0});

    while (!pq.empty() && result.size() < k) {
        auto [v, path, cost] = pq.top();
        pq.pop();
        if (v == dst) {
            result.push_back(path);
            continue;
        }
        if (path.size() >= max_hops) continue;

        for (size_t ei = 0; ei < edges_.size(); ++ei) {
            if (edges_[ei].src_vertex != v) continue;
            // Avoid cycles
            bool cycle = false;
            for (auto e : path) {
                if (edges_[e].dst_vertex == edges_[ei].dst_vertex) {
                    cycle = true;
                    break;
                }
            }
            if (cycle) continue;

            auto new_path = path;
            new_path.push_back(static_cast<uint32_t>(ei));
            double edge_cost = edges_[ei].attrs.latency_us +
                               1e6 / edges_[ei].attrs.bandwidth_gbps;
            pq.push({edges_[ei].dst_vertex, new_path, cost + edge_cost});
        }
    }
    return result;
}

TopologyGraph TopologyGraph::build_h100_node(uint32_t node_id, uint32_t num_gpus) {
    TopologyGraph g;
    std::vector<uint32_t> gpu_verts;

    for (uint32_t i = 0; i < num_gpus; ++i) {
        gpu_verts.push_back(g.add_vertex({node_id, i, EndpointType::GPU}));
    }
    uint32_t cpu = g.add_vertex({node_id, 0, EndpointType::CPU});
    uint32_t cxl = g.add_vertex({node_id, 0, EndpointType::CXLController});
    uint32_t ib = g.add_vertex({node_id, 0, EndpointType::IBPort});

    auto nvlink = default_fabric_attrs(FabricType::NVLink);
    auto pcie = default_fabric_attrs(FabricType::PCIe);
    auto cxl_attrs = default_fabric_attrs(FabricType::CXL);
    auto ib_attrs = default_fabric_attrs(FabricType::InfiniBand);

    // NVLink all-to-all within node
    for (uint32_t i = 0; i < num_gpus; ++i) {
        for (uint32_t j = 0; j < num_gpus; ++j) {
            if (i != j) g.add_edge(gpu_verts[i], gpu_verts[j], nvlink);
        }
    }
    // PCIe GPU-CPU
    for (auto gv : gpu_verts) {
        g.add_edge(gv, cpu, pcie);
        g.add_edge(cpu, gv, pcie);
        g.add_edge(gv, cxl, cxl_attrs);
        g.add_edge(cxl, gv, cxl_attrs);
        g.add_edge(gv, ib, ib_attrs);
        g.add_edge(ib, gv, ib_attrs);
    }
    g.add_edge(cpu, ib, pcie);
    g.add_edge(ib, cpu, pcie);

    return g;
}

TopologyGraph TopologyGraph::build_cluster(uint32_t num_nodes, uint32_t gpus_per_node) {
    TopologyGraph cluster;
    std::vector<std::vector<uint32_t>> node_gpus(num_nodes);
    std::vector<uint32_t> node_ib(num_nodes);

    for (uint32_t n = 0; n < num_nodes; ++n) {
        auto node = build_h100_node(n, gpus_per_node);
        uint32_t base = static_cast<uint32_t>(cluster.vertices().size());
        for (const auto& v : node.vertices()) cluster.add_vertex(v);
        for (const auto& e : node.edges()) {
            cluster.add_edge(e.src_vertex + base, e.dst_vertex + base, e.attrs);
        }
        for (uint32_t g = 0; g < gpus_per_node; ++g) {
            node_gpus[n].push_back(base + g);
        }
        node_ib[n] = base + gpus_per_node + 2;  // GPU×N + CPU + CXL + IB
    }

    auto ib = default_fabric_attrs(FabricType::InfiniBand);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        for (uint32_t j = i + 1; j < num_nodes; ++j) {
            cluster.add_edge(node_ib[i], node_ib[j], ib);
            cluster.add_edge(node_ib[j], node_ib[i], ib);
        }
    }
    return cluster;
}

}  // namespace hics
