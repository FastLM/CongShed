# CongShed — HICS Implementation

**Heterogeneous Interconnect Congestion-aware Scheduler** for LLM serving, as described in the paper *Breaking the Interconnect Bottleneck in LLM Serving via Congestion-Aware Scheduling over Heterogeneous Interconnects*.

HICS is a runtime scheduling layer that:
1. Builds a **latency-weighted topology graph** from hardware telemetry
2. Predicts near-term congestion via a **lightweight LSTM probe**
3. Dynamically steers **KV-cache migrations**, **tensor-parallel activations**, and **pipeline micro-batches** to underutilized links

## Architecture

```
LLM Framework (vLLM / SGLang)
        │ transfer call
        ▼
HICS Interposition Shim  ← NCCL / UCX hooks
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
Topology   LSTM         Path Selection
Graph      Predictor      Engine
   ▲         ▲
   └──── Hardware Telemetry Daemon (1 kHz)
```

## Repository Layout

| Directory | Role |
|-----------|------|
| `cpp/` | Production runtime: NCCL plugin, telemetry daemon, path engine, INT8 LSTM inference |
| `rust/` | Memory-safe alternative runtime with identical scheduling logic |
| `python/` | LSTM training pipeline, topology profiler, serving simulation |

## Core Algorithm

**Latency model (Eq. 1):**

```
λ_ij(s,t) = ℓ_ij + s / (B_ij · (1 - u_ij(t))^α)
```

**Path selection (Algorithm 1):**
1. Compute routing flexibility set `F(t)` for traffic class
2. Enumerate candidate paths (pre-computed k=6 shortest)
3. Select path minimizing `Λ(P, σ, t)` using predicted utilization
4. Preempt KV migrations if TP transfer exceeds deadline

## Build & Run

### C++

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j
./hics_demo
```

NCCL plugin (drop-in via `LD_PRELOAD`):

```bash
export LD_PRELOAD=./libhics_nccl.so
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-70B
```

### Rust

```bash
cd rust
cargo build --release
cargo run --release --bin hics-demo
```

### Python

```bash
cd python
pip install -e .
python examples/simulate_serving.py --lambda-rate 8 --duration 10
```

Train LSTM predictor and export INT8 weights for C++/Rust runtime:

```python
from hics.trainer import CongestionTrainer
trainer = CongestionTrainer.train_and_evaluate(epochs=10, export_path="weights/lstm_int8.bin")
```

## Key Parameters (from paper)

| Parameter | Value |
|-----------|-------|
| LSTM hidden units | 32 |
| History window K | 16 samples @ 1 ms |
| Prediction horizon | 5 ms |
| Telemetry rate | 1 kHz |
| KV chunk size | 256 MB |
| Path enumeration | k=6 (Yen's algorithm) |
| Shim decision budget | < 0.8 µs |
| Contention α (NVLink / IB / PCIe) | 1.4 / 1.8 / 2.1 |

## Integration with vLLM / SGLang

HICS integrates without framework source changes:

- **NCCL**: `ncclNet_v8` plugin registered as preferred transport
- **UCX**: Custom `uct_iface` transport component for KV migrations
- **Startup**: 200 ms profiling sweep populates topology graph
- **Runtime**: Telemetry daemon runs as `SCHED_FIFO` thread

## Component Mapping (Paper → Code)

| Paper Section | C++ | Rust | Python |
|---------------|-----|------|--------|
| §V-B Topology Graph | `topology.hpp` | `topology.rs` | `topology.py` |
| §V-C LSTM Predictor | `lstm_predictor.hpp` | `lstm_predictor.rs` | `lstm_model.py` |
| §V-D Path Selection | `path_engine.hpp` | `path_engine.rs` | `path_engine.py` |
| §V-F SLO Preemption | `path_engine.cpp` | `path_engine.rs` | `path_engine.py` |
| §VI Telemetry | `telemetry.hpp` | `telemetry.rs` | — |
| §VI NCCL Shim | `shim.hpp` | `shim.rs` | — |
| §VI Training | — | — | `trainer.py` |
| §VI Profiler | `shim.cpp` | `shim.rs` | `profiler.py` |

## License

MIT
