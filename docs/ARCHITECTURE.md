# Architecture

This document describes the architecture of nnue-inference-bench.

## Overview

nnue-inference-bench is a standalone benchmark for NNUE neural network inference,
specifically designed for the NikolaChess NKNN v2 model format.

## Components

```
nnue-inference-bench/
├── bench/                  # Benchmark runner
│   ├── runner.py          # Main benchmark CLI
│   ├── results/           # Output JSON files
│   │   ├── LATEST_NNUE.json
│   │   └── raw/           # Timestamped copies
│   └── result_schema.json # JSON schema
├── tools/                  # Core implementation
│   ├── inspect_nknn.py    # NKNN loader + inspector
│   ├── infer_cpu.py       # CPU reference inference
│   ├── infer_gpu.py       # GPU inference (stub)
│   ├── smoke_cpu.py       # CI smoke test
│   └── update_docs.py     # AUTO doc generator
├── native-cuda/            # CUDA kernels (optional)
│   └── nnue_kernels.cu    # Sparse accumulation + dense layers
├── models/                 # NKNN model files
│   └── nikola_d12v2_gold.nknn
├── include/                # Specifications
│   └── nknn_v2.md         # NKNN binary format spec
└── docs/                   # Documentation
    ├── BENCHMARK.md
    └── ARCHITECTURE.md
```

## Data Flow

```
Position → Feature Extraction → Sparse Accumulation → Dense Layers → Eval
   │              │                    │                  │           │
   │        HalfKP encoding       First layer         L2,L3,L4      Score
   │        40960 indices         40960→256           256→32→1
```

## Module Responsibilities

### inspect_nknn.py (Model Loader)

- Parses NKNN v2 binary format
- Validates magic number and version
- Dequantizes weights (cached_property for performance)
- Computes SHA-256 for provenance

### infer_cpu.py (CPU Inference)

- HalfKP feature extraction
- Sparse accumulation (vectorized with NumPy)
- Dense layer forward pass
- SCReLU activation

### infer_gpu.py (GPU Inference)

- GPU detection via nvidia-smi
- Native CUDA kernel loading (ctypes)
- Graceful fallback with documented reason

### runner.py (Benchmark Harness)

- Position generation (deterministic seed)
- Warmup + measured iterations
- Timing with perf_counter_ns / perf_counter
- JSON output with raw data

### update_docs.py (Doc Generator)

- Reads LATEST_NNUE.json
- Updates README.md AUTO sections
- Never recalculates benchmark values

## Neural Network Architecture

```
Input: HalfKP features (40960 per perspective)
       │
       ▼
┌─────────────────────────────────────┐
│ L1 (Sparse): 40960 → 256 per side   │ i16 weights, scale 128
│ Activation: SCReLU                   │
└─────────────────────────────────────┘
       │ Concatenate [white, black] or [black, white] based on STM
       ▼
┌─────────────────────────────────────┐
│ L2 (Dense): 512 → 32                │ i8 weights, scale 64
│ Activation: SCReLU                   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ L3 (Dense): 32 → 32                 │ i8 weights, scale 64
│ Activation: SCReLU                   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ L4 (Dense): 32 → 1                  │ Eval head
└─────────────────────────────────────┘
```

## GPU Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| CUDA kernels | Written | Requires MSVC + CUDA toolkit to compile |
| Python wrapper | Stub | Gracefully reports blocked_reason |
| Benchmark integration | Ready | Falls back to CPU |

To enable GPU:
1. Install CUDA toolkit and Visual Studio
2. Run: `nvcc -shared -o native-cuda/nnue_kernels.dll native-cuda/nnue_kernels.cu`
3. Re-run benchmark

## Design Decisions

1. **Python reference first**: Correctness > performance for baseline
2. **Graceful GPU fallback**: Never crash if GPU unavailable
3. **Canonical JSON**: Single source of truth for all docs
4. **Deterministic checksums**: Prevent benchmark drift
5. **Cached properties**: 1000x speedup for weight access
