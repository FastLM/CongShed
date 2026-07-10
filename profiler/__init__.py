"""Topology profiler: 200 ms startup sweep."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hics.topology import GraphEdge, TopologyGraph
from hics.types import EdgeAttributes, FabricType


@dataclass
class EdgeProfile:
    edge_index: int
    measured_bandwidth_gbps: float
    measured_latency_us: float
    baseline_utilization: float = 0.0


@dataclass
class ClusterProfile:
    num_nodes: int
    gpus_per_node: int
    edge_profiles: List[EdgeProfile] = field(default_factory=list)
    sweep_duration_ms: float = 0.0

    def to_utilization_vector(self) -> List[float]:
        return [p.baseline_utilization for p in self.edge_profiles]


class TopologyProfiler:
    """
    Runs a 200 ms profiling sweep at startup to populate
    baseline bandwidth and latency measurements.
    """

    SWEEP_MS = 200

    def __init__(self, num_nodes: int = 8, gpus_per_node: int = 8):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node

    def build_graph(self) -> TopologyGraph:
        graph = TopologyGraph.build_cluster(self.num_nodes, self.gpus_per_node)
        graph.precompute_paths(k=6, max_hops=5)
        return graph

    def profile_edge(self, edge: GraphEdge) -> EdgeProfile:
        """
        Microbenchmark a single edge. In production, this issues
        loopback transfers and reads hardware counters.
        """
        # Simulated measurements from current edge defaults (override as needed).
        attrs = edge.attrs
        bw = attrs.bandwidth_gbps * (0.95 + 0.05 * (hash(str(edge)) % 10) / 10)
        lat = attrs.latency_us * (0.9 + 0.1 * (hash(str(edge)) % 5) / 5)
        return EdgeProfile(
            edge_index=0,
            measured_bandwidth_gbps=bw,
            measured_latency_us=lat,
            baseline_utilization=0.0,
        )

    def run_sweep(self) -> tuple[TopologyGraph, ClusterProfile]:
        start = time.perf_counter()
        graph = self.build_graph()
        profiles: List[EdgeProfile] = []

        for ei, edge in enumerate(graph.edges):
            profile = self.profile_edge(edge)
            profile.edge_index = ei
            profiles.append(profile)
            # Update graph with measured values
            edge.attrs = EdgeAttributes(
                fabric=edge.attrs.fabric,
                bandwidth_gbps=profile.measured_bandwidth_gbps,
                latency_us=profile.measured_latency_us,
                contention_alpha=edge.attrs.contention_alpha,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        # Pad to SWEEP_MS if faster (simulated wait)
        if elapsed_ms < self.SWEEP_MS:
            time.sleep((self.SWEEP_MS - elapsed_ms) / 1000)

        cluster_profile = ClusterProfile(
            num_nodes=self.num_nodes,
            gpus_per_node=self.gpus_per_node,
            edge_profiles=profiles,
            sweep_duration_ms=max(elapsed_ms, self.SWEEP_MS),
        )
        return graph, cluster_profile

    def fabric_summary(self, graph: TopologyGraph) -> Dict[str, dict]:
        """Summarize profiled fabrics."""
        summary: Dict[str, dict] = {}
        for edge in graph.edges:
            name = edge.attrs.fabric.name
            if name not in summary:
                summary[name] = {
                    "count": 0,
                    "avg_bandwidth_gbps": 0.0,
                    "avg_latency_us": 0.0,
                }
            s = summary[name]
            s["count"] += 1
            s["avg_bandwidth_gbps"] += edge.attrs.bandwidth_gbps
            s["avg_latency_us"] += edge.attrs.latency_us

        for s in summary.values():
            s["avg_bandwidth_gbps"] /= s["count"]
            s["avg_latency_us"] /= s["count"]
        return summary
