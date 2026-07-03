use hics::{EndpointId, EndpointType, HicsShim, TrafficClass};

fn main() {
    let mut shim = HicsShim::new();
    shim.initialize(8, 8);

    println!("HICS Rust demo: 64-GPU cluster");

    shim.telemetry().set_simulated_util(0, 0.95);
    shim.telemetry().set_simulated_util(10, 0.15);

    for _ in 0..1000 {
        shim.intercept_p2p(
            TrafficClass::KVMigration,
            EndpointId {
                node_id: 0,
                local_id: 0,
                endpoint_type: EndpointType::Gpu,
            },
            EndpointId {
                node_id: 4,
                local_id: 0,
                endpoint_type: EndpointType::Gpu,
            },
            256 * 1024 * 1024,
            50_000.0,
        );
    }

    println!("Decisions: {}", shim.decisions_made());
    println!(
        "Avg decision latency: {:.2} µs",
        shim.avg_decision_latency_us()
    );
}
