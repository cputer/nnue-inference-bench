# Contributing to nnue-inference-bench

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install numpy`
3. Run smoke test: `python tools/smoke_cpu.py`
4. Run benchmark: `python bench/runner.py --batch 100 --iters 3`

## Code Structure

- `tools/` - Core Python implementation
- `bench/` - Benchmark runner and results
- `native-cuda/` - CUDA kernels (optional)
- `docs/` - Documentation

## Running Tests

```bash
# Smoke test (CI)
python tools/smoke_cpu.py

# Quick benchmark
python bench/runner.py --batch 100 --warmup 2 --iters 3

# Full benchmark
python bench/runner.py --batch 1000 --warmup 5 --iters 20
```

## Regenerating Documentation

After running benchmarks, update docs from JSON:

```bash
python tools/update_docs.py
```

This updates AUTO sections in README.md from `bench/results/LATEST_NNUE.json`.

## Adding New Features

### Adding a New Kernel

1. Implement in `native-cuda/nnue_kernels.cu`
2. Add wrapper in `tools/infer_gpu.py`
3. Add benchmark case in `bench/runner.py`
4. Update `docs/ARCHITECTURE.md`

### Adding a New Benchmark Tier

1. Add timing function in `bench/runner.py`
2. Update JSON schema in `bench/result_schema.json`
3. Update `tools/update_docs.py` to render new tier
4. Document in `docs/BENCHMARK.md`

## Code Style

- Python: Follow PEP 8
- Use type hints where practical
- Keep functions focused and documented

## Submitting Changes

1. Create a feature branch
2. Run tests: `python tools/smoke_cpu.py`
3. Run benchmark and update docs
4. Verify no uncommitted changes after: `python tools/update_docs.py`
5. Create pull request

## Golden Vector Test

The smoke test uses a fixed input (seed=42) with expected checksum 0xFD4B38E9.
If inference changes, update the expected checksum in `tools/smoke_cpu.py`.
