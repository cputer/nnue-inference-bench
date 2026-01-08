# Benchmark Methodology

This document describes the benchmark tiers and methodology used in nnue-inference-bench.

## Overview

The benchmark measures NNUE neural network inference performance for chess position evaluation.
Two tiers are provided for different measurement granularities.

## Benchmark Tiers

### Tier A: Micro Benchmark (Pure Inference)

Measures only the forward pass computation, excluding all I/O and preprocessing.

- **Scope**: Forward pass kernel only
- **Features**: Pre-extracted before timing
- **Model**: Pre-loaded in memory
- **Metric**: nanoseconds per position (ns/pos)
- **Use case**: Comparing inference implementations

### Tier B: End-to-End Benchmark

Measures the complete pipeline including model loading and feature extraction.

- **Scope**: Full pipeline (load + extract + inference)
- **Model**: Loaded fresh each iteration
- **Metric**: milliseconds per batch (ms/batch), positions per second (pos/s)
- **Use case**: Realistic application performance

## Measurement Protocol

1. **Warmup**: Multiple iterations to stabilize caches and JIT
2. **Measured runs**: N iterations with timing
3. **Statistics**: p50 (median), p95 (tail latency), mean
4. **Checksum**: Deterministic output verification

## Reproducibility

- Fixed random seed for position generation (default: 42)
- Deterministic checksum computed from evaluation outputs
- Raw timing data saved in `bench/results/raw/`

## Device Support

| Device | Status | Notes |
|--------|--------|-------|
| CPU (Python/NumPy) | Available | Reference implementation |
| GPU (CUDA) | Blocked | Requires CUDA toolkit + compiled kernels |

## Running Benchmarks

```bash
# Full benchmark (all tiers, all devices)
python bench/runner.py

# CPU only, Tier B
python bench/runner.py --device cpu --batch 1000 --iters 20

# Quick smoke test
python bench/runner.py --batch 100 --warmup 2 --iters 3
```

## JSON Schema

Results are saved to `bench/results/LATEST_NNUE.json` with this structure:

```json
{
  "meta": {
    "timestamp_utc": "...",
    "model_path": "...",
    "model_sha256": "...",
    "system": {...}
  },
  "config": {
    "batch_size": 1000,
    "warmup_iters": 5,
    "measured_iters": 20,
    "seed": 42
  },
  "results": {
    "device": "cpu",
    "p50_batch_ms": 35.0,
    "p95_batch_ms": 37.0,
    "throughput_pos_per_s": 28000,
    "checksum": "0x6C1B4100"
  }
}
```

## Fairness Rules

- Do not compare Tier A with Tier B (different scopes)
- Always report both p50 and p95 for latency
- Include checksum to verify correctness
- Document system configuration
