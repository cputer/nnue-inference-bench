#!/usr/bin/env python3
"""CPU smoke test for CI."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inspect_nknn import load_nknn
from infer_cpu import create_random_position, extract_halfkp_features, forward_float, compute_checksum

EXPECTED_CHECKSUM = 0xFD4B38E9


def main() -> int:
    repo_root = Path(__file__).parent.parent
    model_path = repo_root / "models" / "nikola_d12v2_gold.nknn"

    print("=" * 50)
    print("NNUE CPU Smoke Test")
    print("=" * 50)

    if not model_path.exists():
        print(f"FAIL: Model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path.name}")
    try:
        model = load_nknn(model_path)
    except Exception as e:
        print(f"FAIL: Model load error: {e}")
        return 1

    print(f"  SHA-256: {model.sha256[:16]}...")
    print(f"  Size: {model.file_size:,} bytes")

    print("Running inference on 10 test positions...")
    evals = []
    for i in range(10):
        pos = create_random_position(42 + i)
        feat_w, feat_b = extract_halfkp_features(pos)
        eval_score, _ = forward_float(model, feat_w, feat_b, pos.stm)
        evals.append(eval_score)

    checksum = compute_checksum(evals)

    print(f"  Checksum: 0x{checksum:08X}")
    print(f"  Expected: 0x{EXPECTED_CHECKSUM:08X}")

    if checksum != EXPECTED_CHECKSUM:
        print("FAIL: Checksum mismatch!")
        return 1

    print()
    print("PASS: All checks passed")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())