// HICS KV-cache migration demo — vLLM / SGLang integration surface.
//
// Demonstrates:
//   1. Dual-rail IB path selection driven by telemetry
//   2. Real KV buffer byte movement (ibverbs backend / host memcpy)
//   3. 256 MB chunk striping across rails
//   4. Chunk-boundary suspension + remaining-chunk reroute under TP pressure
//
// Framework hook (no source changes):
//   export NCCL_NET_PLUGIN=$PWD/libhics_nccl.so
//   # vLLM disagg / SGLang: tag KV transfers with tag>=1000 or size>=256MB
//
// Tag convention (also enforced in shim/nccl_plugin.cpp):
//   tag < 100          → TensorParallel
//   100 <= tag < 1000  → PipelineParallel
//   tag >= 1000        → KVMigration
//   size >= 256 MiB    → KVMigration (overrides tag)

#include "shim/ibverbs_transport.hpp"
#include "shim/nccl_plugin.hpp"
#include "shim/shim.hpp"
#include "shim/ucx_transport.hpp"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;

void fill_pattern(std::vector<uint8_t>& buf, uint8_t seed) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>((seed + i * 131u) & 0xffu);
    }
}

bool verify_prefix(const std::vector<uint8_t>& src, const std::vector<uint8_t>& dst,
                   size_t n) {
    return std::memcmp(src.data(), dst.data(), n) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : nullptr;

    hics::HICSShim shim;
    hics::HICSShim::InitOptions opts;
    opts.num_nodes = 2;
    opts.gpus_per_node = 8;
    if (weights) opts.weights_path = weights;
    if (!shim.initialize(opts)) {
        std::cerr << "HICS init failed\n";
        return 1;
    }

    auto& ib = hics::IbverbsTransport::instance();
    std::cout << "=== HICS KV Migration Demo (vLLM/SGLang surface) ===\n";
    std::cout << "IB backend: " << (ib.hardware() ? "libibverbs/sysfs" : "host-memcpy fallback")
              << "  rails=" << ib.rail_count() << "\n";
    for (const auto& d : ib.devices()) {
        std::cout << "  rail" << d.rail_id << ": " << d.name << " port " << d.port << "\n";
    }
    std::cout << "Telemetry: " << shim.telemetry().active_backend_name() << "\n";

    // --- 1) Real KV buffer transfer with 768 MiB (3 × 256 MiB chunks) ---
    const uint64_t kv_bytes = 768 * kMiB;
    std::vector<uint8_t> src(kv_bytes), dst(kv_bytes, 0);
    fill_pattern(src, 0xA5);

    // Make rail0 look hot so striping prefers rail1 for some chunks
    for (size_t ei = 0; ei < shim.graph().edges().size(); ++ei) {
        const auto& e = shim.graph().edges()[ei];
        if (e.attrs.fabric == hics::FabricType::InfiniBand && e.attrs.rail_id == 0) {
            shim.telemetry().set_simulated_util(static_cast<uint32_t>(ei), 0.92f);
        } else if (e.attrs.fabric == hics::FabricType::InfiniBand && e.attrs.rail_id == 1) {
            shim.telemetry().set_simulated_util(static_cast<uint32_t>(ei), 0.15f);
        }
    }

    auto tid = shim.submit_transfer(
        hics::TrafficClass::KVMigration,
        {0, 0, hics::EndpointType::GPU},
        {1, 0, hics::EndpointType::GPU},
        kv_bytes, 200000.0, src.data(), dst.data());

    auto prog = shim.transfer_status(tid);
    std::cout << "\nStriped KV transfer id=" << tid
              << " chunks=" << prog.chunks.size()
              << " quantum=" << (prog.chunk_quantum / kMiB) << " MiB\n";
    for (const auto& ch : prog.chunks) {
        std::cout << "  chunk" << ch.chunk_seq << " rail=" << ch.rail_id
                  << " offset=" << (ch.byte_offset / kMiB) << " MiB"
                  << " size=" << (ch.chunk_size / kMiB) << " MiB"
                  << " edges=" << ch.edge_indices.size() << "\n";
    }

    // --- 2) Mid-transfer TP pressure → chunk-boundary suspend + reroute ---
    // Small warmup so later chunks stay Pending and get boundary-paused.
    for (int i = 0; i < 2; ++i) shim.poll_transfers(1.0);
    auto mid = shim.transfer_status(tid);
    std::cout << "\nAfter warmup: sent=" << (mid.bytes_sent / kMiB) << " MiB"
              << " state=" << static_cast<int>(mid.state)
              << " chunks_inflight=";
    {
        int n = 0;
        for (const auto& ch : mid.chunks)
            if (ch.state == hics::TransferState::InFlight ||
                ch.state == hics::TransferState::Pending)
                ++n;
        std::cout << n << "\n";
    }

    // Tight TP deadline on an overlapping path triggers preemption
    for (int i = 0; i < 8; ++i) {
        shim.submit_transfer(
            hics::TrafficClass::TensorParallel,
            {0, 0, hics::EndpointType::GPU},
            {1, 0, hics::EndpointType::GPU},
            4 * kMiB, 5.0);
    }

    // Drain through chunk-boundary suspend → telemetry rail reroute → complete
    for (int i = 0; i < 20000; ++i) {
        shim.poll_transfers(1.0);
        auto st = shim.transfer_status(tid);
        if (st.state == hics::TransferState::Completed ||
            st.state == hics::TransferState::Cancelled) {
            break;
        }
    }

    auto done = shim.transfer_status(tid);
    const bool ok = verify_prefix(src, dst, static_cast<size_t>(kv_bytes));
    std::cout << "\nKV complete: state=" << static_cast<int>(done.state)
              << " sent=" << (done.bytes_sent / kMiB) << " MiB"
              << " reroutes=" << done.reroute_count
              << " payload_ok=" << (ok ? "yes" : "NO") << "\n";

    // --- 3) UCX zcopy surface (SGLang-style P2P KV pull) ---
    std::vector<uint8_t> ucx_src(8 * kMiB), ucx_dst(8 * kMiB, 0);
    fill_pattern(ucx_src, 0x3C);
    hics::UcxEndpoint ep;
    shim.ucx().create_ep({0, 1, hics::EndpointType::GPU},
                         {1, 1, hics::EndpointType::GPU}, ep);
    hics::UcxRequest ureq;
    shim.ucx().zcopy_put(ep, ucx_src.data(), ucx_src.size(), ureq,
                         hics::TrafficClass::KVMigration, 5e4, ucx_dst.data());
    while (!shim.ucx().request_completed(ureq)) shim.ucx().progress(1.0);
    const bool ucx_ok = verify_prefix(ucx_src, ucx_dst, ucx_src.size());
    std::cout << "UCX zcopy KV: ok=" << (ucx_ok ? "yes" : "NO") << "\n";

    // --- 4) NCCL plugin dual-device enumeration (vLLM NCCL_NET_PLUGIN) ---
    hics::nccl_plugin_bootstrap(weights);
    int ndev = 0;
    hics::hics_plugin_devices(&ndev);
    std::cout << "NCCL plugin devices (rails): " << ndev << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nStats:\n"
              << "  decisions=" << shim.decisions_made()
              << "  avg_us=" << shim.avg_decision_latency_us() << "\n"
              << "  completed=" << shim.executor().completed_count()
              << "  preempts=" << shim.executor().preempt_count()
              << "  boundary_suspends=" << shim.executor().boundary_suspends()
              << "  reroutes=" << shim.executor().reroute_count() << "\n"
              << "  path_engine preempt/reroute="
              << shim.path_engine().preempt_events() << "/"
              << shim.path_engine().reroute_events() << "\n";

    std::cout << "\nIntegration:\n"
              << "  export NCCL_NET_PLUGIN=./libhics_nccl.so\n"
              << "  # vLLM:  python -m vllm.entrypoints.openai.api_server ...\n"
              << "  # SGLang: use UCX zcopy / NCCL plugin for disagg KV\n"
              << "  # KV tag >= 1000 or size >= 256 MiB → HICS striping\n";

    return (ok && ucx_ok && done.state == hics::TransferState::Completed) ? 0 : 2;
}
