# Benchmark Methodology

This document defines the benchmark tiers, metrics, and reproducibility requirements.

## Benchmark Tiers

### Tier A — Micro (Kernel Only)

Measures pure inference kernel performance, excluding I/O and feature extraction.

- **Input**: Pre-extracted features already in memory
- **Measured**: Forward pass only (sparse accumulation + dense layers)
- **Excludes**: Model loading, feature extraction, position parsing

Use case: Comparing kernel implementations (CUDA vs Mind GPU vs CPU SIMD)

### Tier B — End-to-End

Measures complete inference pipeline including feature extraction.

- **Input**: Board positions (piece placement + side-to-move)
- **Measured**: Feature extraction + forward pass
- **Excludes**: Model loading (amortized across batch)

Use case: Real-world throughput comparison

## Required Metrics

Every benchmark row MUST report:

| Metric | Description | Example |
|--------|-------------|---------|
| `implementation` | Code path being measured | `Python reference`, `Mind CPU`, `CUDA DLL` |
| `device` | Hardware target | `CPU`, `GPU` |
| `tier` | Benchmark tier | `A` or `B` |
| `batch` | Positions per batch | `1000` |
| `warmup` | Warmup iterations | `10` |
| `iters` | Measured iterations | `50` |
| `p50_ms` | Median batch time | `2.245` |
| `p95_ms` | 95th percentile batch time | `2.793` |
| `throughput` | Positions per second | `445345` |
| `checksum` | Deterministic output hash | `0x6C1B4100` |

## Checksum Requirement

All implementations MUST produce identical checksums for the same input positions.

```
Checksum algorithm:
1. For each position, get float32 evaluation score
2. Quantize: int32_value = round(score * 10000)
3. XOR all int32 values together
4. Report as hex: 0x{checksum:08X}
```

If checksums don't match, the implementation is WRONG, not just "different."

## Reproducibility

### Seed Consistency

All benchmarks use `seed=42` for position generation. Same seed = same positions = same checksum.

### Hardware Reporting

Results JSON must include:
- CPU model
- GPU model (if applicable)
- OS and version
- Python/compiler version

### Canonical Output

Results are stored in `bench/results/LATEST_NNUE.json` with full metadata.

## Implementation Status

| Implementation | Tier A | Tier B | Checksum Verified |
|----------------|--------|--------|-------------------|
| Python reference (CPU) | — | ✅ | ✅ 0x6C1B4100 |
| C++ baseline (CPU) | — | ✅ | ✅ 0x6C1B4100 |
| CUDA DLL (GPU) | — | ✅ | ✅ 0x6C1B4100 |
| Mind CPU | — | 🔲 | 🔲 |
| Mind GPU (MIC) | — | 🔲 | 🔲 |

## Winner Determination

The "winner" is determined by **throughput at Tier B** with matching checksum.

```
Winner = max(throughput) WHERE checksum = reference_checksum
```

No checksum match = disqualified.
