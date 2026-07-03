"""Algorithm 1: HICS path selection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hics.lstm_model import CongestionLSTM, HISTORY_LEN, PredictorCache
from hics.topology import TopologyGraph
from hics.types import (
    FabricType,
    PathCost,
    TrafficClass,
    TransferRequest,
    edge_latency_us,
)


@dataclass
class ChunkDispatch:
    chunk_seq: int
    edge_indices: List[int]
    chunk_size: int


@dataclass
class _KvFlow:
    src: int
    dst: int
    bytes_remaining: int
    slack_us: float
    paused: bool = False


class PathSelectionEngine:
    def __init__(
        self,
        graph: TopologyGraph,
        predictor: Optional[CongestionLSTM] = None,
    ):
        self.graph = graph
        self.model = predictor or CongestionLSTM()
        self.cache = PredictorCache(self.model)
        self._utils: List[float] = []
        self._kv_flows: List[_KvFlow] = []

    def flexibility_set(
        self, src: int, dst: int, traffic_class: TrafficClass
    ) -> List[List[int]]:
        """Definition 1: Routing Flexibility Set F(t)."""
        all_paths = self.graph.paths(src, dst)
        admissible: List[List[int]] = []

        for path in all_paths:
            valid = True
            for ei in path:
                fabric = self.graph.edges[ei].attrs.fabric
                if traffic_class == TrafficClass.TENSOR_PARALLEL:
                    if fabric not in (FabricType.NVLINK, FabricType.INFINIBAND):
                        valid = False
                elif traffic_class == TrafficClass.KV_MIGRATION:
                    if fabric == FabricType.PCIE:
                        valid = False
                if not valid:
                    break
            if valid:
                admissible.append(path)

        if not admissible and traffic_class == TrafficClass.KV_MIGRATION:
            return list(all_paths)
        return admissible

    def _path_cost(
        self, edge_indices: List[int], size_bytes: int, transfer_duration_ms: float
    ) -> float:
        total = 0.0
        for ei in edge_indices:
            edge = self.graph.edges[ei]
            util = self._utils[ei] if ei < len(self._utils) else 0.0
            eff = self.cache.effective_util(ei, util, transfer_duration_ms)
            total += edge_latency_us(edge.attrs, size_bytes, eff)
        return total

    def update_predictions(self, utils: List[float]) -> None:
        for e, util in enumerate(utils):
            history = [util] * HISTORY_LEN
            self.cache.update(e, history)

    def select_path(
        self, req: TransferRequest, current_utils: List[float]
    ) -> PathCost:
        """Algorithm 1: HICS Path Selection."""
        self._utils = list(current_utils)

        src = self.graph.vertex_of(req.source)
        dst = self.graph.vertex_of(req.dest)
        if src is None or dst is None:
            return PathCost([], float("inf"))

        candidates = self.flexibility_set(src, dst, req.traffic_class)

        if len(candidates) <= 1:
            if not candidates:
                return PathCost([], float("inf"))
            dur_ms = self._path_cost(candidates[0], req.size_bytes, 1.0) / 1000.0
            return PathCost(
                candidates[0],
                self._path_cost(candidates[0], req.size_bytes, dur_ms),
            )

        est_dur_ms = (req.size_bytes / 1e9 / 50.0) * 1000.0
        best = PathCost([], float("inf"))
        for path in candidates:
            cost = self._path_cost(path, req.size_bytes, est_dur_ms)
            if cost < best.latency_us:
                best = PathCost(path, cost)

        if (
            req.traffic_class == TrafficClass.TENSOR_PARALLEL
            and best.latency_us > req.deadline_us
        ):
            self._preempt_kv_migrations(best.edge_indices)

        return best

    def _preempt_kv_migrations(self, path_edges: List[int]) -> None:
        """SLO-aware preemption (§V-F)."""
        self._kv_flows.sort(key=lambda f: f.slack_us, reverse=True)
        for flow in self._kv_flows:
            if not flow.paused and flow.bytes_remaining > 0:
                flow.paused = True

    def stripe_kv_migration(
        self,
        req: TransferRequest,
        current_utils: List[float],
        chunk_size: int = 256 * 1024 * 1024,
    ) -> List[ChunkDispatch]:
        """Multi-path KV striping with 256 MB chunks (§VI)."""
        dispatches: List[ChunkDispatch] = []
        remaining = req.size_bytes
        seq = 0

        while remaining > 0:
            this_chunk = min(remaining, chunk_size)
            chunk_req = TransferRequest(
                source=req.source,
                dest=req.dest,
                size_bytes=this_chunk,
                traffic_class=req.traffic_class,
                deadline_us=req.deadline_us,
            )
            path = self.select_path(chunk_req, current_utils)
            dispatches.append(
                ChunkDispatch(seq, path.edge_indices, this_chunk)
            )
            seq += 1
            remaining -= this_chunk
        return dispatches
