use crate::lstm_predictor::{LstmPredictor, HISTORY_LEN};
use crate::topology::TopologyGraph;
use crate::types::*;

pub struct PathSelectionEngine {
    graph: TopologyGraph,
    predictor: LstmPredictor,
    utils: Vec<f32>,
    kv_flows: Vec<KvMigrationFlow>,
}

struct KvMigrationFlow {
    #[allow(dead_code)]
    src: u32,
    #[allow(dead_code)]
    dst: u32,
    bytes_remaining: u64,
    slack_us: f64,
    paused: bool,
}

pub struct ChunkDispatch {
    pub chunk_seq: u32,
    pub edge_indices: Vec<u32>,
    pub chunk_size: u64,
}

impl PathSelectionEngine {
    pub fn new(graph: TopologyGraph, predictor: LstmPredictor) -> Self {
        Self {
            graph,
            predictor,
            utils: Vec::new(),
            kv_flows: Vec::new(),
        }
    }

    pub fn graph(&self) -> &TopologyGraph {
        &self.graph
    }

    pub fn predictor_mut(&mut self) -> &mut LstmPredictor {
        &mut self.predictor
    }

    fn flexibility_set(
        &self,
        src: u32,
        dst: u32,
        traffic_class: TrafficClass,
    ) -> Vec<Vec<u32>> {
        let all_paths = self.graph.paths(src, dst);
        let mut admissible = Vec::new();

        for path in all_paths {
            let mut valid = true;
            for &ei in path {
                let edge = &self.graph.edges()[ei as usize];
                match traffic_class {
                    TrafficClass::TensorParallel => {
                        if edge.attrs.fabric != FabricType::NvLink
                            && edge.attrs.fabric != FabricType::InfiniBand
                        {
                            valid = false;
                        }
                    }
                    TrafficClass::KVMigration => {
                        if edge.attrs.fabric == FabricType::Pcie {
                            valid = false;
                        }
                    }
                    TrafficClass::PipelineParallel => {}
                }
                if !valid {
                    break;
                }
            }
            if valid {
                admissible.push(path.clone());
            }
        }

        if admissible.is_empty() && traffic_class == TrafficClass::KVMigration {
            return all_paths;
        }
        admissible
    }

    fn path_cost(&self, edge_indices: &[u32], size_bytes: u64, transfer_duration_ms: f64) -> f64 {
        let mut total = 0.0;
        for &ei in edge_indices {
            let edge = &self.graph.edges()[ei as usize];
            let util = if (ei as usize) < self.utils.len() {
                self.utils[ei as usize] as f64
            } else {
                0.0
            };
            let eff = self
                .predictor
                .effective_util(ei, util, transfer_duration_ms);
            total += edge_latency_us(&edge.attrs, size_bytes, eff);
        }
        total
    }

    /// HICS Path Selection
    pub fn select_path(&mut self, req: &TransferRequest, current_utils: &[f32]) -> PathCost {
        self.utils = current_utils.to_vec();

        let src = match self.graph.vertex_of(&req.source) {
            Some(v) => v,
            None => return PathCost {
                edge_indices: vec![],
                latency_us: f64::MAX,
            },
        };
        let dst = match self.graph.vertex_of(&req.dest) {
            Some(v) => v,
            None => return PathCost {
                edge_indices: vec![],
                latency_us: f64::MAX,
            },
        };

        let candidates = self.flexibility_set(src, dst, req.traffic_class);

        if candidates.len() <= 1 {
            if candidates.is_empty() {
                return PathCost {
                    edge_indices: vec![],
                    latency_us: f64::MAX,
                };
            }
            let dur_ms = self.path_cost(&candidates[0], req.size_bytes, 1.0) / 1000.0;
            return PathCost {
                edge_indices: candidates[0].clone(),
                latency_us: self.path_cost(&candidates[0], req.size_bytes, dur_ms),
            };
        }

        let est_dur_ms = (req.size_bytes as f64 / 1e9 / 50.0) * 1000.0;
        let mut best = PathCost {
            edge_indices: vec![],
            latency_us: f64::MAX,
        };

        for path in &candidates {
            let cost = self.path_cost(path, req.size_bytes, est_dur_ms);
            if cost < best.latency_us {
                best = PathCost {
                    edge_indices: path.clone(),
                    latency_us: cost,
                };
            }
        }

        if req.traffic_class == TrafficClass::TensorParallel && best.latency_us > req.deadline_us
        {
            self.preempt_kv_migrations(&best.edge_indices);
        }

        best
    }

    fn preempt_kv_migrations(&mut self, _path_edges: &[u32]) {
        self.kv_flows.sort_by(|a, b| {
            b.slack_us
                .partial_cmp(&a.slack_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for flow in &mut self.kv_flows {
            if !flow.paused && flow.bytes_remaining > 0 {
                flow.paused = true;
            }
        }
    }

    pub fn stripe_kv_migration(
        &mut self,
        req: &TransferRequest,
        current_utils: &[f32],
        chunk_size: u64,
    ) -> Vec<ChunkDispatch> {
        let mut dispatches = Vec::new();
        let mut remaining = req.size_bytes;
        let mut seq = 0u32;

        while remaining > 0 {
            let this_chunk = remaining.min(chunk_size);
            let chunk_req = TransferRequest {
                size_bytes: this_chunk,
                ..req.clone()
            };
            let path = self.select_path(&chunk_req, current_utils);
            dispatches.push(ChunkDispatch {
                chunk_seq: seq,
                edge_indices: path.edge_indices,
                chunk_size: this_chunk,
            });
            seq += 1;
            remaining -= this_chunk;
        }
        dispatches
    }

    pub fn update_predictions(&mut self, utils: &[f32]) {
        for (e, &util) in utils.iter().enumerate() {
            let mut history = [0.0; HISTORY_LEN];
            history.fill(util as f64);
            let pred = self.predictor.predict(&history);
            self.predictor
                .update_cache(e as u32, util as f64, pred);
        }
    }
}
