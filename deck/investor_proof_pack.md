# Investor Proof Pack
## NNUE Inference Benchmark - Verified Claims

---

## Slide 1: The Proof

### Verified Performance (Checksum: 0x6C1B4100)

| Implementation | Throughput | vs Baseline |
|----------------|-----------|-------------|
| Python NumPy | 35,107 pos/s | reference |
| C++ Scalar | 81,687 pos/s | 1.0x |
| **C++ AVX2** | **469,087 pos/s** | **5.7x** |
| CUDA GPU | 452,714 pos/s | 5.5x |

**Key Result**: At batch=1000, C++ AVX2 matches GPU throughput.

### Why This Matters
- No GPU required for production inference
- Lower power consumption (CPU vs GPU)
- Simpler deployment (no CUDA dependencies)

---

## Slide 2: Honest Limitations

### What This Proves
- AVX2 SIMD delivers 5.7x over scalar C++
- For small batches, CPU can match GPU
- Deterministic: all checksums match

### What This Does NOT Prove
- General GPU inferiority (GPU wins at larger batches)
- Applicability to transformer models
- Transfer overhead is isolated (Tier B includes PCIe)

### Methodology
- Workload: NNUE (40960→256→32→32→1)
- Batch: 1000 positions, seed=42
- 50 iterations, p50 latency
- Hardware: Intel i7-13700H, RTX 4070 Laptop

---

## Verification Commands

```bash
python tools/run_benchmark.py --device both
python tools/validate_results.py --all
python tools/validate_claims.py
```

All results reproducible with checksum 0x6C1B4100.
