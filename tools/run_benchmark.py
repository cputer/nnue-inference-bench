#!/usr/bin/env python3
"""
Unified benchmark runner for all NNUE implementations.
Produces JSON results with checksum verification.

Usage:
    python tools/run_benchmark.py --device cpu    # Python CPU only
    python tools/run_benchmark.py --device gpu    # CUDA GPU only
    python tools/run_benchmark.py --device both   # All implementations
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))
from infer_cpu import load_nknn, Position, extract_halfkp_features, forward_float, compute_checksum, create_random_position

REFERENCE_CHECKSUM = 0x6C1B4100
BATCH_SIZE = 1000
WARMUP_ITERS = 10
MEASURED_ITERS = 50
SEED = 42


def run_python_cpu(model, positions):
    """Run Python CPU benchmark."""
    print("Running Python CPU benchmark...")

    # Pre-extract features
    features = []
    for pos in positions:
        fw, fb = extract_halfkp_features(pos)
        features.append((fw, fb, pos.stm))

    # Warmup
    for _ in range(WARMUP_ITERS):
        for fw, fb, stm in features:
            forward_float(model, fw, fb, stm)

    # Measured
    times_ms = []
    evals = None
    for i in range(MEASURED_ITERS):
        start = time.perf_counter()
        batch_evals = []
        for fw, fb, stm in features:
            e, _ = forward_float(model, fw, fb, stm)
            batch_evals.append(e)
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)
        if i == 0:
            evals = batch_evals

    times_ms.sort()
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    mean = sum(times_ms) / len(times_ms)
    throughput = BATCH_SIZE / (p50 / 1000)
    checksum = compute_checksum(evals)

    return {
        "implementation": "Python reference",
        "device": "CPU",
        "tier": "B",
        "batch_size": BATCH_SIZE,
        "warmup_iters": WARMUP_ITERS,
        "measured_iters": MEASURED_ITERS,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(mean, 3),
        "throughput_pos_per_s": round(throughput, 1),
        "checksum": f"0x{checksum:08X}"
    }


def run_cuda_gpu(model, positions):
    """Run CUDA GPU benchmark."""
    print("Running CUDA GPU benchmark...")

    from infer_gpu import is_gpu_available, get_gpu_status, GPUInference

    status = get_gpu_status()
    if not status["available"]:
        return {"error": f"GPU not available: {status['blocked_reason']}"}

    gpu = GPUInference(model)

    # Pre-extract features
    features_w, features_b, stm_list = [], [], []
    for pos in positions:
        fw, fb = extract_halfkp_features(pos)
        features_w.append(fw)
        features_b.append(fb)
        stm_list.append(pos.stm)

    # Warmup
    for _ in range(WARMUP_ITERS):
        gpu.forward_batch(features_w, features_b, stm_list)

    # Measured
    from infer_gpu import _GPU_STATUS
    times_ms = []
    evals = None
    for i in range(MEASURED_ITERS):
        start = time.perf_counter()
        batch_evals = gpu.forward_batch(features_w, features_b, stm_list)
        _GPU_STATUS["_lib"].nnue_sync()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)
        if i == 0:
            evals = batch_evals.tolist()

    times_ms.sort()
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    mean = sum(times_ms) / len(times_ms)
    throughput = BATCH_SIZE / (p50 / 1000)
    checksum = compute_checksum(evals)

    return {
        "implementation": "CUDA GPU",
        "device": "GPU",
        "gpu_name": status.get("gpu_name", "Unknown"),
        "tier": "B",
        "batch_size": BATCH_SIZE,
        "warmup_iters": WARMUP_ITERS,
        "measured_iters": MEASURED_ITERS,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(mean, 3),
        "throughput_pos_per_s": round(throughput, 1),
        "checksum": f"0x{checksum:08X}"
    }


def run_cpp_simd():
    """Run C++ SIMD (Mind-equivalent) benchmark."""
    print("Running C++ SIMD (C++ AVX2) benchmark...")

    exe = Path(__file__).parent.parent / "native-cpp" / "build" / "bench_simd.exe"
    model = Path(__file__).parent.parent / "models" / "nikola_d12v2_gold.nknn"

    if not exe.exists():
        return {"error": f"C++ SIMD benchmark not built: {exe}"}

    result = subprocess.run(
        [str(exe), "--model", str(model), "--batch", str(BATCH_SIZE),
         "--warmup", str(WARMUP_ITERS), "--iters", str(MEASURED_ITERS), "--seed", str(SEED)],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        return {"error": f"C++ SIMD benchmark failed: {result.stderr}"}

    # Parse JSON output
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Failed to parse output: {result.stdout}"}


def run_cpp_baseline():
    """Run C++ baseline benchmark."""
    print("Running C++ baseline benchmark...")

    exe = Path(__file__).parent.parent / "native-cpp" / "build" / "bench_cpu.exe"
    model = Path(__file__).parent.parent / "models" / "nikola_d12v2_gold.nknn"

    if not exe.exists():
        return {"error": f"C++ baseline not built: {exe}"}

    result = subprocess.run(
        [str(exe), "--model", str(model), "--batch", str(BATCH_SIZE),
         "--warmup", str(WARMUP_ITERS), "--iters", str(MEASURED_ITERS), "--seed", str(SEED)],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        return {"error": f"C++ baseline failed: {result.stderr}"}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Failed to parse output: {result.stdout}"}




def run_mind_cpu():
    """Run Mind CPU benchmark (requires mind compiler)."""
    print("Checking Mind CPU benchmark...")
    
    # Check if mind compiler is available
    import shutil
    mind_exe = shutil.which("mind")
    
    # Also check known install locations
    if not mind_exe:
        known_paths = [
            Path.home() / "projects" / "mind-lang" / "target" / "release" / "mind.exe",
            Path.home() / "projects" / "mind-lang" / "target" / "debug" / "mind.exe",
            Path.home() / ".cargo" / "bin" / "mind.exe",
            Path("C:/Program Files/Mind/bin/mind.exe"),
        ]
        for p in known_paths:
            if p.exists():
                mind_exe = str(p)
                break
    
    if not mind_exe:
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": "mind compiler not found",
            "searched": ["PATH", "~/projects/mind-lang/target/release/", "~/.cargo/bin/"],
            "note": "Build Mind from https://github.com/cputer/mind or add to PATH"
        }
    
    print(f"  Found Mind compiler: {mind_exe}")
    
    # Mind source files
    mind_dir = Path(__file__).parent.parent / "mind-cpu"
    if not mind_dir.exists():
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": "mind-cpu/ directory not found"
        }
    
    # Build Mind binary
    build_dir = mind_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Check Mind compiler capabilities
    try:
        result = subprocess.run([mind_exe, "--help"], capture_output=True, text=True, timeout=10)
        if "eval" in result.stdout and "compile" not in result.stdout.lower():
            return {
                "implementation": "Mind CPU",
                "status": "blocked",
                "blocked_reason": "Mind is interpreter-only (no native compilation yet)",
                "mind_version": subprocess.run([mind_exe, "--version"], capture_output=True, text=True).stdout.strip(),
                "note": "Mind currently supports eval/repl only, native codegen not implemented"
            }
    except Exception as e:
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": f"Mind check failed: {e}"
        }
    
    # If we get here, Mind has compile support - try to compile
    print("Compiling Mind CPU benchmark...")
    try:
        result = subprocess.run(
            [mind_exe, "compile", "-o", str(build_dir / "mind_nnue_bench.exe"),
             str(mind_dir / "bench_main.mind")],
            cwd=str(mind_dir),
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return {
                "implementation": "Mind CPU",
                "status": "blocked", 
                "blocked_reason": f"Compilation failed: {result.stderr}"
            }
    except Exception as e:
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": f"Build error: {e}"
        }
    
    # Run benchmark
    exe = build_dir / "mind_nnue_bench.exe"
    model = Path(__file__).parent.parent / "models" / "nikola_d12v2_gold.nknn"
    
    print("Running Mind CPU benchmark...")
    result = subprocess.run(
        [str(exe), "--model", str(model), "--batch", str(BATCH_SIZE),
         "--warmup", str(WARMUP_ITERS), "--iters", str(MEASURED_ITERS), "--seed", str(SEED)],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": f"Runtime error: {result.stderr}"
        }
    
    try:
        data = json.loads(result.stdout)
        data["status"] = "verified"
        return data
    except json.JSONDecodeError:
        return {
            "implementation": "Mind CPU",
            "status": "blocked",
            "blocked_reason": f"Invalid output: {result.stdout}"
        }

def validate_checksum(results, name, expected=REFERENCE_CHECKSUM):
    """Validate checksum matches reference."""
    if "error" in results:
        return False, results["error"]

    checksum_str = results.get("checksum", "")
    if not checksum_str:
        return False, "No checksum in results"

    checksum = int(checksum_str, 16)
    if checksum != expected:
        return False, f"Checksum mismatch: {checksum_str} != 0x{expected:08X}"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="NNUE Benchmark Runner")
    parser.add_argument("--device", choices=["cpu", "gpu", "both", "all"], default="both",
                       help="Which devices to benchmark")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file (default: bench/results/LATEST_NNUE.json)")
    args = parser.parse_args()

    # Load model
    model_path = Path(__file__).parent.parent / "models" / "nikola_d12v2_gold.nknn"
    print(f"Loading model: {model_path}")
    model = load_nknn(model_path)
    print(f"Model SHA-256: {model.sha256[:16]}...")

    # Generate positions
    print(f"Generating {BATCH_SIZE} positions (seed={SEED})...")
    positions = [create_random_position(SEED + i) for i in range(BATCH_SIZE)]

    results = {
        "generated_utc": datetime.utcnow().isoformat(),
        "reference_checksum": f"0x{REFERENCE_CHECKSUM:08X}",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "results": []
    }

    all_valid = True

    # Run benchmarks
    if args.device in ["cpu", "both", "all"]:
        # Python CPU
        py_results = run_python_cpu(model, positions)
        valid, msg = validate_checksum(py_results, "Python CPU")
        py_results["checksum_valid"] = valid
        results["results"].append(py_results)
        print(f"  Python CPU: {py_results.get('throughput_pos_per_s', 'N/A')} pos/s, checksum {msg}")
        if not valid:
            all_valid = False

        # C++ baseline
        cpp_results = run_cpp_baseline()
        if "error" not in cpp_results:
            valid, msg = validate_checksum(cpp_results, "C++ baseline")
            cpp_results["checksum_valid"] = valid
            results["results"].append(cpp_results)
            print(f"  C++ baseline: {cpp_results.get('throughput_pos_per_s', 'N/A')} pos/s, checksum {msg}")
            if not valid:
                all_valid = False

        # C++ SIMD (C++ AVX2)
        simd_results = run_cpp_simd()
        if "error" not in simd_results:
            valid, msg = validate_checksum(simd_results, "C++ AVX2")
            simd_results["checksum_valid"] = valid
            results["results"].append(simd_results)
            print(f"  C++ AVX2 (SIMD): {simd_results.get('throughput_pos_per_s', 'N/A')} pos/s, checksum {msg}")
            if not valid:
                all_valid = False

        # Mind CPU (requires mind compiler)
        mind_results = run_mind_cpu()
        results["results"].append(mind_results)
        if mind_results.get("status") == "blocked":
            print(f"  Mind CPU: BLOCKED - {mind_results.get('blocked_reason', 'unknown')}")
        elif "error" not in mind_results:
            valid, msg = validate_checksum(mind_results, "Mind CPU")
            mind_results["checksum_valid"] = valid
            print(f"  Mind CPU: {mind_results.get('throughput_pos_per_s', 'N/A')} pos/s, checksum {msg}")
            if not valid:
                all_valid = False

    if args.device in ["gpu", "both", "all"]:
        # CUDA GPU
        gpu_results = run_cuda_gpu(model, positions)
        if "error" not in gpu_results:
            valid, msg = validate_checksum(gpu_results, "CUDA GPU")
            gpu_results["checksum_valid"] = valid
            results["results"].append(gpu_results)
            print(f"  CUDA GPU: {gpu_results.get('throughput_pos_per_s', 'N/A')} pos/s, checksum {msg}")
            if not valid:
                all_valid = False
        else:
            print(f"  CUDA GPU: {gpu_results['error']}")

    results["all_checksums_valid"] = all_valid

    # Save results
    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "bench" / "results" / "LATEST_NNUE.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for r in results["results"]:
        if r.get("status") == "blocked":
            print(f"  {r['implementation']}: BLOCKED - {r.get('blocked_reason', 'unknown')}")
        elif "error" not in r and "throughput_pos_per_s" in r:
            print(f"  {r['implementation']}: {r['throughput_pos_per_s']:,.0f} pos/s (checksum: {r['checksum']})")

    if all_valid:
        print("\nAll checksums match reference: 0x6C1B4100")
        return 0
    else:
        print("\nWARNING: Some checksums did not match!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
