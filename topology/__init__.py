"""Heterogeneous topology graph with k-shortest path precomputation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hics.types import (
    EdgeAttributes,
    EndpointId,
    EndpointType,
    FabricType,
)


@dataclass
class GraphEdge:
    src_vertex: int
    dst_vertex: int
    attrs: EdgeAttributes


class TopologyGraph:
    def __init__(self) -> None:
        self._vertices: List[EndpointId] = []
        self._edges: List[GraphEdge] = []
        self._path_cache: Dict[Tuple[int, int], List[List[int]]] = {}
        self._default_k = 6
        self._default_max_hops = 5

    @property
    def vertices(self) -> List[EndpointId]:
        return self._vertices

    @property
    def edges(self) -> List[GraphEdge]:
        return self._edges

    def add_vertex(self, endpoint: EndpointId) -> int:
        self._vertices.append(endpoint)
        return len(self._vertices) - 1

    def add_edge(self, src: int, dst: int, attrs: EdgeAttributes) -> int:
        self._edges.append(GraphEdge(src, dst, attrs))
        return len(self._edges) - 1

    def vertex_of(self, ep: EndpointId) -> Optional[int]:
        for i, v in enumerate(self._vertices):
            if v == ep:
                return i
        return None

    def paths(self, src: int, dst: int) -> List[List[int]]:
        if (src, dst) not in self._path_cache:
            self._path_cache[(src, dst)] = self._yen_k_shortest(
                src, dst, self._default_k, self._default_max_hops
            )
        return self._path_cache.get((src, dst), [])

    def precompute_paths(self, k: int = 6, max_hops: int = 5) -> None:
        """Store defaults; paths are computed lazily on first access."""
        self._default_k = k
        self._default_max_hops = max_hops

    def _yen_k_shortest(
        self, src: int, dst: int, k: int, max_hops: int
    ) -> List[List[int]]:
        result: List[List[int]] = []
        queue: deque = deque([(src, [], 0.0)])

        while queue and len(result) < k:
            vertex, path, cost = queue.popleft()
            if vertex == dst:
                result.append(path)
                continue
            if len(path) >= max_hops:
                continue
            for ei, edge in enumerate(self._edges):
                if edge.src_vertex != vertex:
                    continue
                if any(
                    self._edges[e].dst_vertex == edge.dst_vertex for e in path
                ):
                    continue
                new_path = path + [ei]
                edge_cost = edge.attrs.latency_us + 1e6 / edge.attrs.bandwidth_gbps
                queue.append((edge.dst_vertex, new_path, cost + edge_cost))
        return result

    @classmethod
    def build_h100_node(cls, node_id: int, num_gpus: int = 8) -> "TopologyGraph":
        g = cls()
        gpu_verts = [
            g.add_vertex(
                EndpointId(node_id, i, EndpointType.GPU)
            )
            for i in range(num_gpus)
        ]
        cpu = g.add_vertex(EndpointId(node_id, 0, EndpointType.CPU))
        cxl = g.add_vertex(EndpointId(node_id, 0, EndpointType.CXL_CONTROLLER))
        ib0 = g.add_vertex(EndpointId(node_id, 0, EndpointType.IB_PORT))
        ib1 = g.add_vertex(EndpointId(node_id, 1, EndpointType.IB_PORT))

        nvlink = FabricType.default_attrs(FabricType.NVLINK)
        pcie = FabricType.default_attrs(FabricType.PCIE)
        cxl_attrs = FabricType.default_attrs(FabricType.CXL)
        ib_r0 = FabricType.default_attrs(FabricType.INFINIBAND, rail_id=0)
        ib_r1 = FabricType.default_attrs(FabricType.INFINIBAND, rail_id=1)

        for i in gpu_verts:
            for j in gpu_verts:
                if i != j:
                    g.add_edge(i, j, nvlink)
        for gv in gpu_verts:
            g.add_edge(gv, cpu, pcie)
            g.add_edge(cpu, gv, pcie)
            g.add_edge(gv, cxl, cxl_attrs)
            g.add_edge(cxl, gv, cxl_attrs)
            g.add_edge(gv, ib0, ib_r0)
            g.add_edge(ib0, gv, ib_r0)
            g.add_edge(gv, ib1, ib_r1)
            g.add_edge(ib1, gv, ib_r1)
        g.add_edge(cpu, ib0, pcie)
        g.add_edge(ib0, cpu, pcie)
        g.add_edge(cpu, ib1, pcie)
        g.add_edge(ib1, cpu, pcie)
        return g

    @classmethod
    def build_cluster(
        cls, num_nodes: int = 8, gpus_per_node: int = 8
    ) -> "TopologyGraph":
        cluster = cls()
        node_ib0: List[int] = []
        node_ib1: List[int] = []

        for n in range(num_nodes):
            node = cls.build_h100_node(n, gpus_per_node)
            base = len(cluster._vertices)
            for v in node.vertices:
                cluster.add_vertex(v)
            for e in node.edges:
                cluster.add_edge(
                    e.src_vertex + base, e.dst_vertex + base, e.attrs
                )
            # GPU×N + CPU + CXL + IB0 + IB1
            node_ib0.append(base + gpus_per_node + 2)
            node_ib1.append(base + gpus_per_node + 3)

        ib_r0 = FabricType.default_attrs(FabricType.INFINIBAND, rail_id=0)
        ib_r1 = FabricType.default_attrs(FabricType.INFINIBAND, rail_id=1)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                cluster.add_edge(node_ib0[i], node_ib0[j], ib_r0)
                cluster.add_edge(node_ib0[j], node_ib0[i], ib_r0)
                cluster.add_edge(node_ib1[i], node_ib1[j], ib_r1)
                cluster.add_edge(node_ib1[j], node_ib1[i], ib_r1)
        return cluster

    def to_networkx(self):
        """Export for visualization."""
        import networkx as nx

        g = nx.DiGraph()
        for i, v in enumerate(self._vertices):
            g.add_node(i, label=str(v))
        for ei, e in enumerate(self._edges):
            g.add_edge(
                e.src_vertex,
                e.dst_vertex,
                key=ei,
                fabric=e.attrs.fabric.name,
                bandwidth=e.attrs.bandwidth_gbps,
            )
        return g
