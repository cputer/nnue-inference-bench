#!/usr/bin/env python3
"""
NNUE Inference Benchmark Runner.
"""

from __future__ import annotations
import json
import time
import platform
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from inspect_nknn import load_nknn, NKNNModel
from infer_cpu import (
    Position, extract_halfkp_features, forward_float,
    create_random_position, compute_checksum
)


def generate_test_positions(count: int, seed: int = 42) -> List[Position]:
    return [create_random_position(seed + i) for i in range(count)]


def run_inference_batch(model: NKNNModel, positions: List[Position]) -> List[float]:
    evals = []
    for pos in positions:
        feat_w, feat_b = extract_halfkp_features(pos)
        eval_score, _ = forward_float(model, feat_w, feat_b, pos.stm)
        evals.append(eval_score)
    return evals


def benchmark_cpu(
    model: NKNNModel,
    positions: List[Position],
    warmup_iters: int,
    measured_iters: int
) -> Dict[str, Any]:
    print(f"  Warmup: {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _ = run_inference_batch(model, positions)

    print(f"  Measured: {measured_iters} iterations...")
    times_ms = []
    all_evals = []

    for i in range(measured_iters):
        start = time.perf_counter()
        evals = run_inference_batch(model, positions)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)
        if i == 0:
            all_evals = evals

    times_arr = np.array(times_ms)
    p50 = float(np.percentile(times_arr, 50))
    p95 = float(np.percentile(times_arr, 95))
    mean_ms = float(np.mean(times_arr))

    positions_per_batch = len(positions)
    throughput = (positions_per_batch / (p50 / 1000)) if p50 > 0 else 0

    checksum = compute_checksum(all_evals)

    return {
        "times_ms": times_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean_ms,
        "throughput_pos_per_s": throughput,
        "checksum": checksum,
        "positions_per_batch": positions_per_batch,
    }


def detect_gpu():
    from infer_gpu import get_gpu_status
    status = get_gpu_status()
    return {
        "available": status.get("available", False),
        "name": status.get("gpu_name"),
        "blocked_reason": status.get("blocked_reason"),
    }


def benchmark_gpu(model, positions, warmup_iters, measured_iters):
    """Run GPU benchmark."""
    from infer_gpu import GPUInference, _GPU_STATUS
    from infer_cpu import extract_halfkp_features

    gpu = GPUInference(model)
    
    # Pre-extract features
    features_w, features_b, stm_list = [], [], []
    for pos in positions:
        fw, fb = extract_halfkp_features(pos)
        features_w.append(fw)
        features_b.append(fb)
        stm_list.append(pos.stm)

    print(f"  Warmup: {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _ = gpu.forward_batch(features_w, features_b, stm_list)
        _GPU_STATUS["_lib"].nnue_sync()

    print(f"  Measured: {measured_iters} iterations...")
    times_ms = []
    all_evals = None

    for i in range(measured_iters):
        start = time.perf_counter()
        evals = gpu.forward_batch(features_w, features_b, stm_list)
        _GPU_STATUS["_lib"].nnue_sync()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
        if i == 0:
            all_evals = list(evals)

    times_arr = np.array(times_ms)
    p50 = float(np.percentile(times_arr, 50))
    p95 = float(np.percentile(times_arr, 95))
    mean_ms = float(np.mean(times_arr))
    throughput = (len(positions) / (p50 / 1000)) if p50 > 0 else 0
    checksum = compute_checksum(all_evals)

    return {
        "times_ms": times_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean_ms,
        "throughput_pos_per_s": throughput,
        "checksum": checksum,
        "positions_per_batch": len(positions),
    }

def get_git_commit():
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except:
        return "unknown"

def get_system_info():
    gpu = detect_gpu()
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "git_commit": get_git_commit(),
        "gpu_name": gpu["name"],
        "gpu_available": gpu["available"],
        "gpu_blocked_reason": gpu["blocked_reason"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NNUE Inference Benchmark")
    ap.add_argument("--model", "-m", default="models/nikola_d12v2_gold.nknn")
    ap.add_argument("--batch", "-b", type=int, default=1000)
    ap.add_argument("--warmup", "-w", type=int, default=5)
    ap.add_argument("--iters", "-i", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--device", "-d", choices=["cpu", "gpu", "both"], default="both")
    args = ap.parse_args()

    repo_root = Path(__file__).parent.parent
    model_path = repo_root / args.model

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / "bench" / "results" / "LATEST_NNUE.json"

    print(f"Loading model: {model_path}")
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    model = load_nknn(model_path)
    print(f"  SHA-256: {model.sha256}")
    print(f"  Size: {model.file_size:,} bytes")
    print()

    print(f"Generating {args.batch} test positions (seed={args.seed})...")
    positions = generate_test_positions(args.batch, args.seed)
    print()

    print("Running CPU benchmark...")
    results = benchmark_cpu(model, positions, args.warmup, args.iters)
    print()

    timestamp = datetime.now(timezone.utc).isoformat()

    output = {
        "meta": {
            "timestamp_utc": timestamp,
            "model_path": str(model_path.relative_to(repo_root)),
            "model_sha256": model.sha256,
            "model_size_bytes": model.file_size,
            "system": get_system_info(),
        },
        "config": {
            "batch_size": args.batch,
            "warmup_iters": args.warmup,
            "measured_iters": args.iters,
            "seed": args.seed,
        },
        "results": {
            "device": "cpu",
            "p50_batch_ms": round(results["p50_ms"], 3),
            "p95_batch_ms": round(results["p95_ms"], 3),
            "mean_batch_ms": round(results["mean_ms"], 3),
            "throughput_pos_per_s": round(results["throughput_pos_per_s"], 1),
            "checksum": f"0x{results['checksum']:08X}",
        },
        "raw_times_ms": [round(t, 3) for t in results["times_ms"]],
    }

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device:     {output['results']['device']}")
    print(f"Batch size: {args.batch} positions")
    print(f"Iterations: {args.iters} (measured), {args.warmup} (warmup)")
    print()
    print(f"p50 batch time:  {output['results']['p50_batch_ms']:.3f} ms")
    print(f"p95 batch time:  {output['results']['p95_batch_ms']:.3f} ms")
    print(f"Throughput:      {output['results']['throughput_pos_per_s']:,.0f} pos/s")
    print(f"Checksum:        {output['results']['checksum']}")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    raw_dir = output_path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp_safe = timestamp.replace(":", "-").replace("+", "_")
    raw_path = raw_dir / f"cpu_{timestamp_safe}.json"
    raw_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Raw log saved to: {raw_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())