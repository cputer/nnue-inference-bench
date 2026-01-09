#!/usr/bin/env python3
"""
Validates benchmark results JSON against reference checksum.
This is a CI gate - exits with non-zero if validation fails.

Usage:
    python tools/validate_results.py bench/results/LATEST_NNUE.json
    python tools/validate_results.py --all  # Validate all results in bench/results/
"""

import json
import sys
import argparse
from pathlib import Path

REFERENCE_CHECKSUM = "0x6C1B4100"


def validate_result(result: dict) -> tuple:
    """Validate a single benchmark result."""
    if not isinstance(result, dict):
        return False, f"Invalid result format: {type(result)}"
    
    impl = result.get("implementation", "Unknown")
    checksum = result.get("checksum", "")

    if "error" in result:
        return False, f"{impl}: Error - {result['error']}"

    if not checksum:
        return False, f"{impl}: No checksum found"

    # Normalize checksum format
    checksum = checksum.upper()
    ref = REFERENCE_CHECKSUM.upper()

    if checksum != ref:
        return False, f"{impl}: Checksum mismatch ({checksum} != {ref})"

    throughput = result.get("throughput_pos_per_s", 0)
    if throughput <= 0:
        return False, f"{impl}: Invalid throughput ({throughput})"

    return True, f"{impl}: OK ({throughput:,.0f} pos/s)"


def validate_file(filepath: Path) -> tuple:
    """Validate all results in a JSON file."""
    messages = []
    all_valid = True

    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return False, [f"Failed to load {filepath}: {e}"]

    # Handle multiple JSON formats
    results = []
    
    if isinstance(data, list):
        results = data
    elif "results" in data:
        results = data["results"]
    elif "checksum" in data and "implementation" in data:
        # Single result format
        results = [data]
    elif "cpu_results" in data or "gpu_results" in data:
        # Old format with cpu_results/gpu_results
        if "cpu_results" in data:
            results.append({
                "implementation": "Python CPU",
                "checksum": data["cpu_results"].get("checksum", ""),
                "throughput_pos_per_s": data["cpu_results"].get("throughput_pos_per_s", 0)
            })
        if "gpu_results" in data:
            results.append({
                "implementation": "CUDA GPU",
                "checksum": data["gpu_results"].get("checksum", ""),
                "throughput_pos_per_s": data["gpu_results"].get("throughput_pos_per_s", 0)
            })
    else:
        return False, [f"Unknown format in {filepath}"]

    if not results:
        return False, [f"No results found in {filepath}"]

    for result in results:
        if isinstance(result, dict):
            valid, msg = validate_result(result)
            messages.append(msg)
            if not valid:
                all_valid = False

    return all_valid, messages


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark results")
    parser.add_argument("files", nargs="*", help="JSON files to validate")
    parser.add_argument("--all", action="store_true", help="Validate all results in bench/results/")
    args = parser.parse_args()

    files = []
    if args.all:
        results_dir = Path(__file__).parent.parent / "bench" / "results"
        files = list(results_dir.glob("*.json"))
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        parser.print_help()
        return 1

    if not files:
        print("No JSON files found to validate")
        return 1

    print(f"Validating {len(files)} file(s) against reference checksum {REFERENCE_CHECKSUM}")
    print()

    all_passed = True

    for filepath in files:
        print(f"=== {filepath.name} ===")
        valid, messages = validate_file(filepath)
        for msg in messages:
            status = "PASS" if "OK" in msg else "FAIL"
            print(f"  [{status}] {msg}")
        if not valid:
            all_passed = False
        print()

    if all_passed:
        print("All validations PASSED")
        return 0
    else:
        print("Some validations FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
