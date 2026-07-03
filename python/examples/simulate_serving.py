#!/usr/bin/env python3
"""
End-to-end HICS serving simulation.

Simulates disaggregated prefill/decode with TP/PP/KV traffic classes
under Poisson arrivals and compares static vs HICS routing.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List

from hics.path_engine import PathSelectionEngine
from hics.profiler import TopologyProfiler
from hics.trainer import CongestionTrainer, UtilizationDataset, generate_synthetic_trace
from hics.types import (
    EndpointId,
    EndpointType,
    PathCost,
    TrafficClass,
    TransferRequest,
)


@dataclass
class Request:
    req_id: int
    prompt_tokens: int
    arrival_ms: float
    ttft_ms: float = 0.0


@dataclass
class SimConfig:
    num_nodes: int = 8
    gpus_per_node: int = 8
    lambda_rate: float = 8.0  # req/s
    duration_s: float = 60.0
    mean_prompt_tokens: int = 512
    use_hics: bool = True


class ServingSimulator:
    KV_BYTES_PER_TOKEN = 80 * 2 * 8 * 128 * 2  # Llama-3-70B FP16

    def __init__(self, config: SimConfig):
        self.config = config
        profiler = TopologyProfiler(config.num_nodes, config.gpus_per_node)
        self.graph, self.profile = profiler.run_sweep()

        trace = generate_synthetic_trace(int(config.duration_s * 1000) + 5000)
        trainer = CongestionTrainer()
        ds = UtilizationDataset(trace[:8000])
        for _ in range(2):
            trainer.train_epoch(ds, batch_size=128)
        self.engine = PathSelectionEngine(self.graph, trainer.model)

        self.link_utils: List[float] = [0.0] * len(self.graph.edges)

    def _inject_congestion(self, t_ms: float) -> None:
        """Simulate time-varying link load."""
        hour = (t_ms / 3600_000) % 24
        peak_factor = 1.3 if 10 <= hour <= 18 else 0.8
        for i in range(len(self.link_utils)):
            base = random.uniform(0.2, 0.5) * peak_factor
            if random.random() < 0.05:
                base = random.uniform(0.85, 0.99)
            self.link_utils[i] = min(base, 0.99)

    def _route_static_nvlink(
        self,
        src_node: int,
        dst_node: int,
        size_bytes: int,
    ) -> PathCost:
        """Static-NVLink baseline: always prefer NVLink paths, ignore congestion."""
        src = EndpointId(src_node, 0, EndpointType.GPU)
        dst = EndpointId(dst_node, 0, EndpointType.GPU)
        s = self.graph.vertex_of(src)
        d = self.graph.vertex_of(dst)
        if s is None or d is None:
            return PathCost([], float("inf"))

        from hics.types import edge_latency_us, FabricType

        paths = self.graph.paths(s, d)
        nvlink_paths = [
            p for p in paths
            if all(self.graph.edges[ei].attrs.fabric == FabricType.NVLINK for ei in p)
        ]
        chosen = nvlink_paths[0] if nvlink_paths else (paths[0] if paths else [])
        if not chosen:
            return PathCost([], float("inf"))

        # Static routing ignores alternative paths even when congested
        util = max(self.link_utils[ei] for ei in chosen) if chosen else 0.5
        cost = sum(
            edge_latency_us(self.graph.edges[ei].attrs, size_bytes, util)
            for ei in chosen
        )
        return PathCost(chosen, cost)

    def _route(
        self,
        src_node: int,
        dst_node: int,
        size_bytes: int,
        traffic_class: TrafficClass,
        static_nvlink: bool,
    ) -> PathCost:
        src = EndpointId(src_node, 0, EndpointType.GPU)
        dst = EndpointId(dst_node, 0, EndpointType.GPU)
        req = TransferRequest(
            source=src,
            dest=dst,
            size_bytes=size_bytes,
            traffic_class=traffic_class,
            deadline_us=100.0 if traffic_class == TrafficClass.TENSOR_PARALLEL else 50_000.0,
        )

        if static_nvlink or not self.config.use_hics:
            return self._route_static_nvlink(src_node, dst_node, size_bytes)
        self.engine.update_predictions(self.link_utils)
        return self.engine.select_path(req, self.link_utils)

    def simulate_request(self, req: Request, static_nvlink: bool) -> float:
        t = req.arrival_ms
        self._inject_congestion(t)

        prefill_ms = req.prompt_tokens * 0.03  # ~30 µs/token compute
        kv_size = req.prompt_tokens * self.KV_BYTES_PER_TOKEN

        src_prefill = random.randint(0, 3)
        dst_decode = random.randint(4, 7)

        kv_cost = self._route(
            src_prefill, dst_decode, kv_size, TrafficClass.KV_MIGRATION, static_nvlink
        )
        tp_cost = self._route(
            src_prefill, src_prefill, 16 * 1024 * 1024,
            TrafficClass.TENSOR_PARALLEL, static_nvlink,
        )

        migration_ms = kv_cost.latency_us / 1000.0
        tp_ms = tp_cost.latency_us / 1000.0
        queue_ms = random.expovariate(1.0 / 50.0) if static_nvlink else random.expovariate(1.0 / 120.0)

        return prefill_ms + migration_ms + tp_ms + queue_ms

    def run(self, static_nvlink: bool = False) -> List[Request]:
        cfg = self.config
        requests: List[Request] = []
        t = 0.0
        req_id = 0
        while t < cfg.duration_s * 1000:
            inter_arrival = random.expovariate(cfg.lambda_rate / 1000.0)
            t += inter_arrival
            prompt = max(64, int(random.gauss(cfg.mean_prompt_tokens, 128)))
            req = Request(req_id, prompt, t)
            req.ttft_ms = self.simulate_request(req, static_nvlink)
            requests.append(req)
            req_id += 1
        return requests


def percentile(data: List[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main():
    parser = argparse.ArgumentParser(description="HICS serving simulation")
    parser.add_argument("--lambda-rate", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--no-hics", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig(lambda_rate=args.lambda_rate, duration_s=args.duration)

    print("=== Static-NVLink baseline ===")
    static_reqs = ServingSimulator(cfg).run(static_nvlink=True)
    static_ttfts = [r.ttft_ms for r in static_reqs]
    print(f"  Requests: {len(static_reqs)}")
    print(f"  P50 TTFT: {percentile(static_ttfts, 50):.0f} ms")
    print(f"  P99 TTFT: {percentile(static_ttfts, 99):.0f} ms")

    print("\n=== HICS ===")
    hics_cfg = SimConfig(
        lambda_rate=args.lambda_rate,
        duration_s=args.duration,
        use_hics=not args.no_hics,
    )
    hics_reqs = ServingSimulator(hics_cfg).run(static_nvlink=False)
    hics_ttfts = [r.ttft_ms for r in hics_reqs]
    print(f"  Requests: {len(hics_reqs)}")
    print(f"  P50 TTFT: {percentile(hics_ttfts, 50):.0f} ms")
    print(f"  P99 TTFT: {percentile(hics_ttfts, 99):.0f} ms")

    p99_static = percentile(static_ttfts, 99)
    p99_hics = percentile(hics_ttfts, 99)
    if p99_static > 0:
        reduction = (1 - p99_hics / p99_static) * 100
        print(f"\nP99 TTFT reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
