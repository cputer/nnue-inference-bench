# nnue-inference-bench

A **standalone NNUE inference benchmark** for NikolaChess neural network evaluation.

## Features

- Loads NKNN v2 binary model format (see [include/nknn_v2.md](include/nknn_v2.md))
- CPU reference inference with deterministic checksums
- Benchmark harness with warmup, timing, and percentiles
- Canonical JSON output for CI/docs automation

## Quick Start

```bash
# Inspect model
python tools/inspect_nknn.py --model models/nikola_d12v2_gold.nknn

# Run CPU benchmark
python bench/runner.py --batch 1000 --warmup 5 --iters 20

# Quick smoke test
python tools/smoke_cpu.py
```

## Latest Benchmark Results

<!-- AUTO:BENCHMARK_START -->
| Device | Batch | p50 (ms) | p95 (ms) | Throughput | Checksum |
|--------|-------|----------|----------|------------|----------|
| CPU | 1000 | 35.603 | 38.219 | 28,088 pos/s | 0x6C1B4100 |

*Updated: 2026-01-08 | Model: nikola_d12v2_gold.nknn*
<!-- AUTO:BENCHMARK_END -->

## GPU Status

<!-- AUTO:GPU_STATUS_START -->
| Device | Status | Reason |
|--------|--------|--------|
| NVIDIA GeForce RTX 4070 Laptop GPU | Blocked | CUDA kernels not compiled |
<!-- AUTO:GPU_STATUS_END -->

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for GPU enablement instructions.

## Model

**Gold model**: `models/nikola_d12v2_gold.nknn`
- SHA-256: `041BC71A527B830E54A83664BD7B2C4414A9F047348943DF2CE3359B49142E3C`
- Architecture: 40960 -> 256 -> 32 -> 32 -> 1 (+WDL)
- Size: 20,989,768 bytes

See [include/nknn_v2.md](include/nknn_v2.md) for full binary format specification.

## Repository Layout

```
nnue-inference-bench/
|-- bench/              # Benchmark runner and results
|   |-- runner.py       # Main benchmark CLI
|   |-- results/        # JSON outputs (LATEST.json, raw/)
|   +-- result_schema.json
|-- tools/              # Core implementation
|   |-- inspect_nknn.py # NKNN loader
|   |-- infer_cpu.py    # CPU inference
|   |-- infer_gpu.py    # GPU inference (stub)
|   |-- smoke_cpu.py    # CI smoke test
|   +-- update_docs.py  # AUTO doc generator
|-- native-cuda/        # CUDA kernels (optional)
|-- docs/               # Documentation
|-- models/             # NKNN model files
|-- data/               # Test position data
|-- include/            # Format specifications
|   +-- nknn_v2.md      # NKNN v2 binary format spec
+-- .github/workflows/  # CI configuration
```

## Benchmark Tiers

- **Tier A (Micro)**: Pure inference kernel timing, no I/O
- **Tier B (End-to-end)**: Full pipeline including load + features + inference

Current implementation: Tier B (Python reference)

## Development

### Running Tests

```bash
python tools/smoke_cpu.py
```

### Updating Docs from JSON

```bash
python tools/update_docs.py
```

## License

MIT
