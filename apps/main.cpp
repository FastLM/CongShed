#include "shim/shim.hpp"
#include "shim/ucx_transport.hpp"

#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    const char* weights = nullptr;
    if (argc > 1) weights = argv[1];

    hics::HICSShim shim;
    hics::HICSShim::InitOptions opts;
    opts.num_nodes = 8;
    opts.gpus_per_node = 8;
    if (weights) opts.weights_path = weights;

    if (!shim.initialize(opts)) {
        std::cerr << "HICS initialization failed\n";
        return 1;
    }

    std::cout << "HICS initialized: 64-GPU cluster topology\n";
    std::cout << "Telemetry backend: " << shim.telemetry().active_backend_name()
              << (shim.telemetry().using_hardware() ? " (hw)" : " (synthetic)") << "\n";
    std::cout << "LSTM weights: " << (shim.weights_loaded() ? "loaded" : "defaults") << "\n";

    // Sample TP path cost
    auto utils = std::vector<float>(shim.graph().edges().size(), 0.3f);
    auto cost = shim.path_engine().select_path(
        {{0, 0, hics::EndpointType::GPU},
         {0, 1, hics::EndpointType::GPU},
         16 * 1024 * 1024,
         hics::TrafficClass::TensorParallel,
         100.0},
        utils);
    std::cout << "Sample TP path cost: " << cost.latency_us << " µs\n";

    // Congested NVLink + KV on same GPUs, then tight TP that must preempt
    shim.telemetry().set_simulated_util(0, 0.95f);
    shim.telemetry().set_simulated_util(1, 0.90f);

    for (int i = 0; i < 64; ++i) {
        // Intra-node KV shares NVLink with upcoming TP
        shim.submit_transfer(
            hics::TrafficClass::KVMigration,
            {0, 0, hics::EndpointType::GPU},
            {0, 1, hics::EndpointType::GPU},
            256 * 1024 * 1024,
            50000.0);
    }

    // Tight TP deadline on the same pair triggers KV preemption + reroute
    for (int i = 0; i < 32; ++i) {
        shim.submit_transfer(
            hics::TrafficClass::TensorParallel,
            {0, 0, hics::EndpointType::GPU},
            {0, 1, hics::EndpointType::GPU},
            4 * 1024 * 1024,
            5.0);
    }

    for (int i = 0; i < 200; ++i) shim.poll_transfers(1.0);

    // UCX zcopy path for a KV chunk
    hics::UcxEndpoint ep;
    shim.ucx().create_ep({0, 0, hics::EndpointType::GPU},
                         {1, 0, hics::EndpointType::GPU}, ep);
    hics::UcxRequest ureq;
    std::vector<char> buf(8 * 1024 * 1024);
    shim.ucx().zcopy_put(ep, buf.data(), buf.size(), ureq);
    while (!shim.ucx().request_completed(ureq)) shim.ucx().progress(1.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Decisions: " << shim.decisions_made() << "\n";
    std::cout << "Avg decision latency: " << shim.avg_decision_latency_us() << " µs\n";
    std::cout << "Transfers completed: " << shim.executor().completed_count() << "\n";
    std::cout << "Preempts: " << shim.executor().preempt_count()
              << "  Reroutes: " << shim.executor().reroute_count() << "\n";
    std::cout << "Bytes transferred: " << shim.executor().bytes_transferred() << "\n";
    std::cout << "Path-engine preempt/reroute events: "
              << shim.path_engine().preempt_events() << "/"
              << shim.path_engine().reroute_events() << "\n";
    return 0;
}
