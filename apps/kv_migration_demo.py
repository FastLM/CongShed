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
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# In-tree package-dir layout (same as pyproject.toml) without pip install
_hics = types.ModuleType("hics")
_hics.__path__ = [_ROOT]  # mark as package
sys.modules["hics"] = _hics

import model as _model  # noqa: E402

sys.modules["hics.types"] = _model
_hics.types = _model

import topology as _topology  # noqa: E402

sys.modules["hics.topology"] = _topology
_hics.topology = _topology

import lstm as _lstm  # noqa: E402

sys.modules["hics.lstm"] = _lstm
_hics.lstm = _lstm

import path_engine as _path_engine  # noqa: E402

sys.modules["hics.path_engine"] = _path_engine

from path_engine import PathSelectionEngine  # noqa: E402
from topology import TopologyGraph  # noqa: E402
from model import (  # noqa: E402
    EndpointId,
    EndpointType,
    TrafficClass,
    TransferRequest,
)

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

    graph = TopologyGraph.build_cluster(num_nodes=2, gpus_per_node=8)
    graph.precompute_paths()
    engine = PathSelectionEngine(graph)

    utils = [0.3] * len(graph.edges)
    for i, e in enumerate(graph.edges):
        if e.attrs.fabric.name == "INFINIBAND":
            utils[i] = 0.9 if e.attrs.rail_id == 0 else 0.1

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
    print("NCCL_NET_PLUGIN tip: export NCCL_NET_PLUGIN=./libhics_nccl.so")
    print(f"dual-rail IB edges present: "
          f"{sum(1 for e in graph.edges if e.attrs.rail_id >= 0)}")

    chunks = engine.stripe_kv_migration(req, utils, chunk_size=KV_CHUNK)
    print(f"striped chunks: {len(chunks)} × ≤256 MiB")
    for c in chunks:
        print(f"  chunk {c.chunk_seq}: {c.chunk_size / (1024*1024):.0f} MiB "
              f"rail={c.rail_id} offset={c.byte_offset / (1024*1024):.0f} MiB")

    print("\nFramework wiring:")
    print("  vLLM  — load plugin via NCCL_NET_PLUGIN; KV blocks ≥256MiB auto-classed")
    print("  SGLang — disagg transfer: tag≥1000 or UCX zcopy through HICS transport")
    print("  C++    — ./build/kv_migration_demo  (real buffers + boundary suspend)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
