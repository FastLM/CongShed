#!/usr/bin/env python3
"""
vLLM / SGLang KV migration integration demo (Python side).

Shows the tag/size conventions HICS uses when loaded as NCCL_NET_PLUGIN,
and walks a simulated disaggregated prefill→decode KV handoff with
256 MiB chunk striping + telemetry-biased rail selection.

For the full dual-rail + real buffer path, run the C++ demo:
  ./build/kv_migration_demo
"""

from __future__ import annotations

import argparse
import os
import sys

# Prefer in-repo package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hics.path_engine import PathSelectionEngine  # noqa: E402
from hics.profiler import TopologyProfiler  # noqa: E402
from hics.types import EndpointId, EndpointType, TrafficClass, TransferRequest  # noqa: E402

KV_CHUNK = 256 * 1024 * 1024

# Must stay aligned with shim/nccl_plugin.cpp::classify_tag
TAG_TP = 1
TAG_PP = 100
TAG_KV = 1000


def classify(tag: int, size: int) -> TrafficClass:
    if size >= KV_CHUNK or tag >= TAG_KV:
        return TrafficClass.KV_MIGRATION
    if tag >= TAG_PP:
        return TrafficClass.PIPELINE_PARALLEL
    return TrafficClass.TENSOR_PARALLEL


def main() -> int:
    ap = argparse.ArgumentParser(description="HICS KV migration integration demo")
    ap.add_argument("--kv-mib", type=int, default=512, help="KV payload size in MiB")
    args = ap.parse_args()

    profiler = TopologyProfiler(num_nodes=2, gpus_per_node=8)
    graph, _ = profiler.run_sweep()
    engine = PathSelectionEngine(graph)

    utils = [0.3] * len(graph.edges)
    # Bias rail-like IB edges hotter on even indices (proxy for rail0)
    for i, e in enumerate(graph.edges):
        if getattr(e.attrs, "fabric", None) is not None:
            name = str(e.attrs.fabric)
            if "InfiniBand" in name or "IB" in name:
                utils[i] = 0.9 if (i % 2 == 0) else 0.1

    kv_bytes = args.kv_mib * 1024 * 1024
    req = TransferRequest(
        source=EndpointId(0, 0, EndpointType.GPU),
        dest=EndpointId(1, 0, EndpointType.GPU),
        size_bytes=kv_bytes,
        traffic_class=classify(TAG_KV, kv_bytes),
        deadline_us=200_000.0,
    )

    print("=== HICS KV migration demo (Python / framework conventions) ===")
    print(f"classify(tag={TAG_KV}, size={args.kv_mib}MiB) → {req.traffic_class}")
    print(f"NCCL_NET_PLUGIN tip: export NCCL_NET_PLUGIN=./libhics_nccl.so")

    if hasattr(engine, "stripe_kv_migration"):
        chunks = engine.stripe_kv_migration(req, utils, chunk_size=KV_CHUNK)
        print(f"striped chunks: {len(chunks)} × ≤256 MiB")
        for c in chunks:
            seq = getattr(c, "chunk_seq", c.get("chunk_seq") if isinstance(c, dict) else "?")
            size = getattr(c, "chunk_size", c.get("chunk_size") if isinstance(c, dict) else 0)
            print(f"  chunk {seq}: {size / (1024*1024):.0f} MiB")
    else:
        path = engine.select_path(req, utils)
        print(f"select_path latency_us={path.latency_us:.1f} edges={len(path.edge_indices)}")

    print("\nFramework wiring:")
    print("  vLLM  — load plugin via NCCL_NET_PLUGIN; KV blocks ≥256MiB auto-classed")
    print("  SGLang — disagg transfer: tag≥1000 or UCX zcopy through HICS transport")
    print("  C++    — ./build/kv_migration_demo  (real buffers + boundary suspend)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
