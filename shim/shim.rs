use crate::lstm_predictor::LstmPredictor;
use crate::path_engine::PathSelectionEngine;
use crate::telemetry::TelemetryDaemon;
use crate::topology::TopologyGraph;
use crate::types::*;
use std::time::Instant;

pub struct HicsShim {
    telemetry: TelemetryDaemon,
    path_engine: PathSelectionEngine,
    decisions_made: u64,
    total_decision_ns: u64,
}

impl HicsShim {
    pub fn new() -> Self {
        Self {
            telemetry: TelemetryDaemon::new(0),
            path_engine: PathSelectionEngine::new(TopologyGraph::new(), LstmPredictor::new()),
            decisions_made: 0,
            total_decision_ns: 0,
        }
    }

    pub fn initialize(&mut self, num_nodes: u32, gpus_per_node: u32) -> bool {
        let mut graph = TopologyGraph::build_cluster(num_nodes, gpus_per_node);
        graph.precompute_paths(6, 5);
        let num_edges = graph.edges().len();

        self.telemetry = TelemetryDaemon::new(num_edges);
        self.path_engine = PathSelectionEngine::new(graph, LstmPredictor::new());
        self.run_profiling_sweep();
        self.telemetry.start();
        true
    }

    fn run_profiling_sweep(&self) {
        let start = Instant::now();
        while start.elapsed().as_millis() < 200 {}
    }

    pub fn telemetry(&self) -> &TelemetryDaemon {
        &self.telemetry
    }

    pub fn path_engine_mut(&mut self) -> &mut PathSelectionEngine {
        &mut self.path_engine
    }

    pub fn dispatch_transfer(&mut self, req: &TransferRequest) -> PathCost {
        let t0 = Instant::now();
        let ring = self.telemetry.ring();
        let num_edges = self.path_engine.graph().edges().len();
        let utils = ring.snapshot(num_edges);
        self.path_engine.update_predictions(&utils);

        let result = if req.traffic_class == TrafficClass::KVMigration
            && req.size_bytes > 1024 * 1024 * 1024
        {
            let chunks = self
                .path_engine
                .stripe_kv_migration(req, &utils, 256 * 1024 * 1024);
            let mut total = PathCost {
                edge_indices: chunks
                    .first()
                    .map(|c| c.edge_indices.clone())
                    .unwrap_or_default(),
                latency_us: 0.0,
            };
            for c in chunks {
                let chunk_req = TransferRequest {
                    size_bytes: c.chunk_size,
                    ..req.clone()
                };
                total.latency_us += self.path_engine.select_path(&chunk_req, &utils).latency_us;
            }
            total
        } else {
            self.path_engine.select_path(req, &utils)
        };

        self.decisions_made += 1;
        self.total_decision_ns += t0.elapsed().as_nanos() as u64;
        result
    }

    pub fn intercept_p2p(
        &mut self,
        cls: TrafficClass,
        src: EndpointId,
        dst: EndpointId,
        size_bytes: u64,
        deadline_us: f64,
    ) -> i32 {
        let req = TransferRequest {
            source: src,
            dest: dst,
            size_bytes,
            traffic_class: cls,
            deadline_us,
        };
        let _ = self.dispatch_transfer(&req);
        0
    }

    pub fn decisions_made(&self) -> u64 {
        self.decisions_made
    }

    pub fn avg_decision_latency_us(&self) -> f64 {
        if self.decisions_made == 0 {
            return 0.0;
        }
        self.total_decision_ns as f64 / self.decisions_made as f64 / 1000.0
    }
}

impl Default for HicsShim {
    fn default() -> Self {
        Self::new()
    }
}
