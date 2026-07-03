use crate::types::*;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub src_vertex: u32,
    pub dst_vertex: u32,
    pub attrs: EdgeAttributes,
}

pub struct TopologyGraph {
    vertices: Vec<EndpointId>,
    edges: Vec<GraphEdge>,
    path_cache: HashMap<(u32, u32), Vec<Vec<u32>>>,
}

impl TopologyGraph {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            path_cache: HashMap::new(),
        }
    }

    pub fn add_vertex(&mut self, endpoint: EndpointId) -> u32 {
        self.vertices.push(endpoint);
        (self.vertices.len() - 1) as u32
    }

    pub fn add_edge(&mut self, src: u32, dst: u32, attrs: EdgeAttributes) -> u32 {
        self.edges.push(GraphEdge {
            src_vertex: src,
            dst_vertex: dst,
            attrs,
        });
        (self.edges.len() - 1) as u32
    }

    pub fn vertices(&self) -> &[EndpointId] {
        &self.vertices
    }

    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    pub fn vertex_of(&self, ep: &EndpointId) -> Option<u32> {
        self.vertices
            .iter()
            .position(|v| v == ep)
            .map(|i| i as u32)
    }

    pub fn paths(&self, src: u32, dst: u32) -> &[Vec<u32>] {
        static EMPTY: Vec<Vec<u32>> = Vec::new();
        self.path_cache
            .get(&(src, dst))
            .map(|v| v.as_slice())
            .unwrap_or(&EMPTY)
    }

    pub fn precompute_paths(&mut self, k: usize, max_hops: usize) {
        self.path_cache.clear();
        let n = self.vertices.len();
        for src in 0..n {
            for dst in 0..n {
                if src == dst {
                    continue;
                }
                let paths = self.yen_k_shortest(src as u32, dst as u32, k, max_hops);
                self.path_cache.insert((src as u32, dst as u32), paths);
            }
        }
    }

    fn yen_k_shortest(&self, src: u32, dst: u32, k: usize, max_hops: usize) -> Vec<Vec<u32>> {
        #[derive(Clone)]
        struct State {
            vertex: u32,
            path: Vec<u32>,
            cost: f64,
        }

        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(State {
            vertex: src,
            path: vec![],
            cost: 0.0,
        });

        while let Some(state) = queue.pop_front() {
            if result.len() >= k {
                break;
            }
            if state.vertex == dst {
                result.push(state.path);
                continue;
            }
            if state.path.len() >= max_hops {
                continue;
            }

            for (ei, edge) in self.edges.iter().enumerate() {
                if edge.src_vertex != state.vertex {
                    continue;
                }
                let cycle = state.path.iter().any(|&e| {
                    self.edges[e as usize].dst_vertex == edge.dst_vertex
                });
                if cycle {
                    continue;
                }
                let mut new_path = state.path.clone();
                new_path.push(ei as u32);
                let edge_cost = edge.attrs.latency_us + 1e6 / edge.attrs.bandwidth_gbps;
                queue.push_back(State {
                    vertex: edge.dst_vertex,
                    path: new_path,
                    cost: state.cost + edge_cost,
                });
            }
        }
        result
    }

    pub fn build_h100_node(node_id: u32, num_gpus: u32) -> Self {
        let mut g = Self::new();
        let gpu_verts: Vec<u32> = (0..num_gpus)
            .map(|i| {
                g.add_vertex(EndpointId {
                    node_id,
                    local_id: i,
                    endpoint_type: EndpointType::Gpu,
                })
            })
            .collect();

        let cpu = g.add_vertex(EndpointId {
            node_id,
            local_id: 0,
            endpoint_type: EndpointType::Cpu,
        });
        let cxl = g.add_vertex(EndpointId {
            node_id,
            local_id: 0,
            endpoint_type: EndpointType::CxlController,
        });
        let ib = g.add_vertex(EndpointId {
            node_id,
            local_id: 0,
            endpoint_type: EndpointType::IbPort,
        });

        let nvlink = FabricType::NvLink.default_attrs();
        let pcie = FabricType::Pcie.default_attrs();
        let cxl_attrs = FabricType::Cxl.default_attrs();
        let ib_attrs = FabricType::InfiniBand.default_attrs();

        for &i in &gpu_verts {
            for &j in &gpu_verts {
                if i != j {
                    g.add_edge(i, j, nvlink);
                }
            }
        }
        for &gv in &gpu_verts {
            g.add_edge(gv, cpu, pcie);
            g.add_edge(cpu, gv, pcie);
            g.add_edge(gv, cxl, cxl_attrs);
            g.add_edge(cxl, gv, cxl_attrs);
            g.add_edge(gv, ib, ib_attrs);
            g.add_edge(ib, gv, ib_attrs);
        }
        g.add_edge(cpu, ib, pcie);
        g.add_edge(ib, cpu, pcie);
        g
    }

    pub fn build_cluster(num_nodes: u32, gpus_per_node: u32) -> Self {
        let mut cluster = Self::new();
        let mut node_ib = Vec::new();

        for n in 0..num_nodes {
            let node = Self::build_h100_node(n, gpus_per_node);
            let base = cluster.vertices.len() as u32;
            for v in node.vertices {
                cluster.add_vertex(v);
            }
            for e in node.edges {
                cluster.add_edge(
                    e.src_vertex + base,
                    e.dst_vertex + base,
                    e.attrs,
                );
            }
            node_ib.push(base + gpus_per_node + 2);
        }

        let ib = FabricType::InfiniBand.default_attrs();
        for i in 0..num_nodes as usize {
            for j in (i + 1)..num_nodes as usize {
                cluster.add_edge(node_ib[i], node_ib[j], ib);
                cluster.add_edge(node_ib[j], node_ib[i], ib);
            }
        }
        cluster
    }
}

impl Default for TopologyGraph {
    fn default() -> Self {
        Self::new()
    }
}
