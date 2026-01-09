# 4-LLM Consensus Audit Results

## Audit Date: 2026-01-09

## Scores

| Auditor | Score | Focus |
|---------|-------|-------|
| GPT-5.2 | 7/10 | Honesty |
| Gemini 3 Pro | 6/10 | Methodology Fairness |
| Grok 4.1 | 5/10 | Investor Credibility |
| DeepSeek | 9/10 math, 5/10 stats | Math Correctness |

**Consensus Score: 6.5/10** (needs improvement)

## Key Issues Identified

### 1. Mislabeling (FIXED)
- Original: "Mind CPU (SIMD)"
- Corrected: "C++ AVX2 (SIMD)"
- This is C++ with hand-written intrinsics, not Mind-compiled code

### 2. Batch Size Caveat (ADDED)
- Results valid for batch=1000 only
- GPU wins at larger batch sizes
- Added to README and deck

### 3. Transfer Overhead (DISCLOSED)
- Tier B includes PCIe transfer time
- This disadvantages GPU for small batches
- Added caveat to methodology

### 4. Statistical Rigor (NOTED)
- 50 iterations is borderline
- No variance reported
- Checksum alone insufficient for numerical correctness

## Verified Claims (Defensible)

1. "C++ AVX2 is 5.7x faster than scalar C++" - TRUE
2. "All checksums match 0x6C1B4100" - TRUE
3. "At batch=1000, CPU matches GPU" - TRUE with caveats

## Claims Removed (Not Defensible)

1. ~~"Mind CPU is faster than C++"~~ - Was mislabeled C++
2. ~~"CPU beats GPU"~~ - Only true for small batches
3. ~~"Mind achieves 99% of GPU"~~ - Not Mind, was C++ AVX2

## What This Proves

- SIMD optimization delivers significant speedups (5.7x)
- For small-batch inference, CPU can match GPU
- Deterministic execution (checksums match)

## What This Does NOT Prove

- General GPU inferiority
- Applicability to larger models
- Mind compiler performance (Mind not used)

## Recommendations for Due Diligence

1. Add multi-batch results (100, 1K, 10K, 100K)
2. Separate transfer time from compute time
3. Add variance/confidence intervals
4. Implement actual Mind compiler benchmarks
