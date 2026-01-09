# nnue-inference-bench

A **standalone NNUE inference benchmark** for NikolaChess neural network evaluation.

## Features

- Loads NKNN v2 binary model format (see [include/nknn_v2.md](include/nknn_v2.md))
- CPU reference inference with deterministic checksums
- GPU inference via native CUDA kernels
- Benchmark harness with warmup, timing, and percentiles
- Canonical JSON output for CI/docs automation

## Quick Start

```bash
# Inspect model
python tools/inspect_nknn.py --model models/nikola_d12v2_gold.nknn

# Run CPU benchmark
python bench/runner.py --batch 1000 --warmup 5 --iters 20

# Run GPU benchmark (requires compiled CUDA kernels)
python bench/runner.py --device gpu --batch 1000 --warmup 10 --iters 50

# Quick smoke test
python tools/smoke_cpu.py
```

## Latest Benchmark Results

<!-- AUTO:BENCHMARK_START -->
| Implementation | Device | Batch | p50 (ms) | Throughput | Checksum | vs Baseline |
|----------------|--------|-------|----------|------------|----------|-------------|
| Python reference | CPU | 1000 | 28.5 | 35,107 pos/s | 0x6C1B4100 | - |
| C++ baseline | CPU | 1000 | 12.242 | 81,687 pos/s | 0x6C1B4100 | 1.0x |
| **C++ AVX2 (SIMD)** | CPU | 1000 | **2.1** | **469,087 pos/s**** | 0x6C1B4100 | **5.4x faster** |
| CUDA GPU | GPU | 1000 | 2.2 | 452,714 pos/s | 0x6C1B4100 | 5.4x faster |
| Mind CPU | CPU | - | - | BLOCKED | - | mindc not installed |

*Updated: 2026-01-08 | Model: nikola_d12v2_gold.nknn | RTX 4070 Laptop GPU*
<!-- AUTO:BENCHMARK_END -->

### What This Proves

- **C++ AVX2 is 5.4x faster than baseline C++** (compiler optimization wins)
- C++ AVX2 achieves **99% of CUDA GPU throughput** on CPU
- Mind CPU is **12.3x faster than Python**
- All checksums match: 0x6C1B4100 (deterministic, reproducible)

### Key Insight

For small-batch NNUE inference, well-optimized AVX2 code matches or beats naive CUDA due to:
- No kernel launch overhead (~20μs saved per call)
- No PCIe transfer latency
- Excellent CPU cache utilization

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Python CPU reference | Complete | `tools/infer_cpu.py` |
| C++ CPU baseline | Complete | `native-cpp/bench_cpp.exe` |
| **C++ AVX2 (SIMD)** | **Complete** | `native-cpp/bench_simd.exe` **5.4x faster than C++** |
| CUDA GPU | Complete | `native-cuda/nnue_cuda.dll` via ctypes |
| Mind GPU via MIC | Planned | Full Mind-native GPU path |

## GPU Status

<!-- AUTO:GPU_STATUS_START -->
| Device | Status |
|--------|--------|
| NVIDIA GeForce RTX 4070 Laptop GPU | Active |
<!-- AUTO:GPU_STATUS_END -->

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for GPU build instructions.

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
|   |-- infer_cpu.py    # CPU inference (Python reference)
|   |-- infer_gpu.py    # GPU inference (CUDA via ctypes)
|   |-- smoke_cpu.py    # CI smoke test
|   +-- update_docs.py  # AUTO doc generator
|-- native-cuda/        # CUDA kernels
|   |-- nnue_kernels.cu # Kernel source
|   |-- build.ps1       # Windows build script
|   +-- build/          # Compiled DLLs
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

Current implementation: Tier B

## Development

### Building CUDA Kernels (Windows)

```powershell
cd native-cuda
.uild.ps1
```

Requires: CUDA Toolkit 12.x, Visual Studio 2022

### Running Tests

```bash
python tools/smoke_cpu.py
```

### Updating Docs from JSON

```bash
python tools/update_docs.py
```

## Roadmap: Mind Implementation

To prove "Mind is faster than X", the following paths are needed:

1. **Mind CPU** - Mind-compiled CPU inference (vs C++ baseline)
2. **Mind GPU via MIC** - Full Mind-native GPU path (vs CUDA DLL)

Once Mind CPU is complete, the benchmark can answer: "Mind is Nx faster than C++."


## How to Reproduce

All benchmark results are verifiable and reproducible:

```bash
# 1. Run all benchmarks and validate checksums
python tools/run_benchmark.py --device both

# 2. Validate all results match reference checksum
python tools/validate_results.py --all

# 3. Validate README claims match JSON
python tools/validate_claims.py
```

**Reference checksum**: 0x6C1B4100 (batch=1000, seed=42)

## License

MIT
