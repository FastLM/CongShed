# CongShed Implementation

**Heterogeneous Interconnect Congestion-aware Scheduler** for LLM serving.

HICS is a runtime scheduling layer that:
1. Builds a **latency-weighted topology graph** from hardware telemetry
2. Predicts near-term congestion via a **lightweight LSTM probe**
3. Dynamically steers **KV-cache migrations**, **tensor-parallel activations**, and **pipeline micro-batches** to underutilized links — including **dual IB rails**

## Architecture

```
LLM Framework (vLLM / SGLang)
        │ transfer call
        ▼
HICS Interposition Shim  ← NCCL (2 rails) / UCX zcopy / ibverbs
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
Topology   LSTM         Path Selection
Graph      Predictor      Engine
   ▲         ▲
   └──── Hardware Telemetry Daemon (1 kHz, SCHED_FIFO)
```

## Repository Layout

| Module | Role |
|--------|------|
| `model/` | Shared domain model (endpoints, fabrics, 256 MB KV chunk) |
| `topology/` | Dual-rail latency-weighted graph + Yen's k-shortest paths |
| `lstm/` | INT8 inference (C++/Rust) + PyTorch training & weight export |
| `path_engine/` | Telemetry-driven routing, KV striping, SLO preemption |
| `telemetry/` | 1 kHz ring-buffer daemon (NVML / IB MAD / PMU + synthetic) |
| `transfer/` | Per-chunk execution, boundary suspend, rail reroute |
| `shim/` | NCCL `ncclNet_v8` plugin + UCX zcopy + ibverbs backend |
| `profiler/` | Startup topology profiling (Python) |
| `apps/` | Demos: `hics_demo`, `kv_migration_demo`, serving simulation |
| `hics/` | Python package entry (`pip install -e .`) |
| `src/` | Rust crate root (`lib.rs`) |

## Core Algorithm

**Latency model:**

```
λ_ij(s,t) = ℓ_ij + s / (B_ij · (1 - u_ij(t))^α)
```

**Path selection:**
1. Compute routing flexibility set `F(t)` for traffic class
2. Enumerate candidate paths (pre-computed k=6 shortest) across **dual IB rails**
3. Select path minimizing `Λ(P, σ, t)` using predicted utilization
4. Stripe KV migrations into **256 MB chunks**; suspend at chunk boundaries and reroute remaining chunks when TP misses SLO

## Build & Run

### C++

```bash
mkdir -p build && cd build
cmake ..
make -j
./hics_demo                        # optional: ./hics_demo ../weights/lstm_int8.bin
./kv_migration_demo                # dual-rail + real KV buffers + boundary suspend
```

NCCL plugin (`ncclNetPlugin_v8` in `libhics_nccl.so`, **2 devices = 2 rails**):

```bash
export NCCL_NET_PLUGIN=./libhics_nccl.so
# Tag convention: tag>=1000 or size>=256MiB → KVMigration
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-70B
```

### Rust

```bash
cargo build --release
cargo run --release --bin hics-demo
```

### Python

```bash
pip install -e .
python apps/simulate_serving.py --lambda-rate 8 --duration 10
python apps/kv_migration_demo.py --kv-mib 512
```

Train predictor and export INT8 weights for C++/Rust runtime:

```python
from hics.lstm import CongestionTrainer
trainer = CongestionTrainer.train_and_evaluate(epochs=10, export_path="weights/lstm_int8.bin")
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| LSTM hidden units | 32 |
| History window K | 16 samples @ 1 ms |
| Prediction horizon | 5 ms |
| Telemetry rate | 1 kHz |
| KV chunk size | 256 MB |
| IB rails | 2 (`mlx5_0` / `mlx5_1`) |
| Path enumeration | k=6 (Yen's algorithm) |
| Shim decision budget | < 0.8 µs |
| Contention α (NVLink / IB / PCIe) | 1.4 / 1.8 / 2.1 |

## Integration with vLLM / SGLang

HICS integrates without framework source changes:

- **NCCL**: `ncclNet_v8` plugin exposes **two net devices** (rail0 / rail1); load via `NCCL_NET_PLUGIN`
- **UCX**: `UcxTransport::zcopy_put/get` moves real KV buffers through the dual-rail ibverbs backend
- **Tags**: `tag < 100` TP · `100–999` PP · `≥1000` or `size≥256MiB` KV
- **Startup**: 200 ms profiling sweep populates topology graph
- **Runtime**: Telemetry daemon runs as `SCHED_FIFO` thread (best-effort)
- **Demo**: `./kv_migration_demo` and `apps/kv_migration_demo.py`

## Component Mapping

| Component | C++ | Rust | Python |
|-----------|-----|------|--------|
| Topology Graph | `topology/topology.hpp` | `topology/topology.rs` | `topology/` |
| LSTM Predictor | `lstm/lstm_predictor.hpp` | `lstm/lstm_predictor.rs` | `lstm/lstm_model.py` |
| Path Selection | `path_engine/path_engine.hpp` | `path_engine/path_engine.rs` | `path_engine/` |
| Transfer Exec | `transfer/transfer_executor.hpp` | — | — |
| Telemetry | `telemetry/` (NVML, IB MAD, uncore PMU) | `telemetry/telemetry.rs` | — |
| NCCL Plugin | `shim/nccl_plugin.*` + `cuda_mr.*` | — | — |
| IB / UCX Transport | `shim/ibverbs_transport.*` + `ucx_transport.*` | — | — |
| Training | — | — | `lstm/trainer.py` |
| Profiler | `shim/shim.cpp` | `shim/shim.rs` | `profiler/` |

## License

MIT
