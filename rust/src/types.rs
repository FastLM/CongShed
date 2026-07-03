use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FabricType {
    NvLink,
    Pcie,
    Cxl,
    InfiniBand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrafficClass {
    TensorParallel,
    PipelineParallel,
    KVMigration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndpointType {
    Gpu,
    Cpu,
    CxlController,
    IbPort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointId {
    pub node_id: u32,
    pub local_id: u32,
    pub endpoint_type: EndpointType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EdgeAttributes {
    pub fabric: FabricType,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
    pub contention_alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRequest {
    pub source: EndpointId,
    pub dest: EndpointId,
    pub size_bytes: u64,
    pub traffic_class: TrafficClass,
    pub deadline_us: f64,
}

#[derive(Debug, Clone)]
pub struct PathCost {
    pub edge_indices: Vec<u32>,
    pub latency_us: f64,
}

impl FabricType {
    pub fn contention_exponent(self) -> f64 {
        match self {
            FabricType::NvLink => 1.4,
            FabricType::InfiniBand => 1.8,
            FabricType::Pcie | FabricType::Cxl => 2.1,
        }
    }

    pub fn default_attrs(self) -> EdgeAttributes {
        match self {
            FabricType::NvLink => EdgeAttributes {
                fabric: self,
                bandwidth_gbps: 900.0,
                latency_us: 1.0,
                contention_alpha: 1.4,
            },
            FabricType::Pcie => EdgeAttributes {
                fabric: self,
                bandwidth_gbps: 64.0,
                latency_us: 3.0,
                contention_alpha: 2.1,
            },
            FabricType::Cxl => EdgeAttributes {
                fabric: self,
                bandwidth_gbps: 50.0,
                latency_us: 5.0,
                contention_alpha: 2.1,
            },
            FabricType::InfiniBand => EdgeAttributes {
                fabric: self,
                bandwidth_gbps: 50.0,
                latency_us: 5.0,
                contention_alpha: 1.8,
            },
        }
    }
}

impl fmt::Display for FabricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FabricType::NvLink => write!(f, "NVLink"),
            FabricType::Pcie => write!(f, "PCIe"),
            FabricType::Cxl => write!(f, "CXL"),
            FabricType::InfiniBand => write!(f, "InfiniBand"),
        }
    }
}

/// λ_ij(s,t) = ℓ_ij + s / (B_ij · (1 - u_ij(t))^α)
pub fn edge_latency_us(attrs: &EdgeAttributes, size_bytes: u64, utilization: f64) -> f64 {
    let u = utilization.clamp(0.0, 0.99);
    let residual = 1.0 - u;
    let transfer_s = (size_bytes as f64 / 1e9)
        / (attrs.bandwidth_gbps * residual.powf(attrs.contention_alpha));
    attrs.latency_us + transfer_s * 1e6
}
