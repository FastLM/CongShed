"""Core types and latency model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class FabricType(Enum):
    NVLINK = auto()
    PCIE = auto()
    CXL = auto()
    INFINIBAND = auto()

    @property
    def contention_alpha(self) -> float:
        return {
            FabricType.NVLINK: 1.4,
            FabricType.INFINIBAND: 1.8,
            FabricType.PCIE: 2.1,
            FabricType.CXL: 2.1,
        }[self]

    @classmethod
    def default_attrs(cls, fabric: "FabricType") -> "EdgeAttributes":
        defaults = {
            FabricType.NVLINK: (900.0, 1.0),
            FabricType.PCIE: (64.0, 3.0),
            FabricType.CXL: (50.0, 5.0),
            FabricType.INFINIBAND: (50.0, 5.0),
        }
        bw, lat = defaults[fabric]
        return EdgeAttributes(
            fabric=fabric,
            bandwidth_gbps=bw,
            latency_us=lat,
            contention_alpha=fabric.contention_alpha,
        )


class TrafficClass(Enum):
    TENSOR_PARALLEL = auto()
    PIPELINE_PARALLEL = auto()
    KV_MIGRATION = auto()


class EndpointType(Enum):
    GPU = auto()
    CPU = auto()
    CXL_CONTROLLER = auto()
    IB_PORT = auto()


@dataclass(frozen=True)
class EndpointId:
    node_id: int
    local_id: int
    endpoint_type: EndpointType


@dataclass
class EdgeAttributes:
    fabric: FabricType
    bandwidth_gbps: float
    latency_us: float
    contention_alpha: float


@dataclass
class TransferRequest:
    source: EndpointId
    dest: EndpointId
    size_bytes: int
    traffic_class: TrafficClass
    deadline_us: float


@dataclass
class PathCost:
    edge_indices: List[int]
    latency_us: float


def edge_latency_us(
    attrs: EdgeAttributes, size_bytes: int, utilization: float
) -> float:
    """λ_ij(s,t) = ℓ_ij + s / (B_ij · (1 - u_ij(t))^α)"""
    u = min(max(utilization, 0.0), 0.99)
    residual = 1.0 - u
    transfer_s = (size_bytes / 1e9) / (
        attrs.bandwidth_gbps * (residual ** attrs.contention_alpha)
    )
    return attrs.latency_us + transfer_s * 1e6
