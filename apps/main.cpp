#include "shim/shim.hpp"

#include <iostream>
#include <iomanip>

int main() {
    hics::HICSShim shim;
    if (!shim.initialize(8, 8)) {
        std::cerr << "HICS initialization failed\n";
        return 1;
    }

    std::cout << "HICS initialized: 64-GPU cluster topology\n";
    std::cout << "Edges: " << shim.path_engine().select_path(
        {{0, 0, hics::EndpointType::GPU},
         {1, 0, hics::EndpointType::GPU},
         16 * 1024 * 1024,
         hics::TrafficClass::TensorParallel,
         100.0},
        std::vector<float>(512, 0.5f)).latency_us << " µs (sample TP path cost)\n";

    // Simulate KV migration under congestion
    shim.telemetry().set_simulated_util(0, 0.95f);
    shim.telemetry().set_simulated_util(10, 0.15f);

    for (int i = 0; i < 1000; ++i) {
        shim.intercept_p2p(
            hics::TrafficClass::KVMigration,
            {0, 0, hics::EndpointType::GPU},
            {4, 0, hics::EndpointType::GPU},
            256 * 1024 * 1024,
            50000.0);
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Decisions: " << shim.decisions_made() << "\n";
    std::cout << "Avg decision latency: " << shim.avg_decision_latency_us() << " µs\n";
    return 0;
}
