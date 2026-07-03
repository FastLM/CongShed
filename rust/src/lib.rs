//! HICS — Heterogeneous Interconnect Congestion-aware Scheduler
//!
//! Rust implementation mirroring the C++ runtime described in the paper.

pub mod lstm_predictor;
pub mod path_engine;
pub mod telemetry;
pub mod topology;
pub mod types;

pub use path_engine::PathSelectionEngine;
pub use shim::HicsShim;
pub use topology::TopologyGraph;
pub use types::*;

mod shim;
