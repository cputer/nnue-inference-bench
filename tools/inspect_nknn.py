#!/usr/bin/env python3
"""
NKNN v2 Model Inspector and Loader.

Parses NKNN binary format and prints structure/stats.
Also provides NKNNModel class for use by inference code.
"""

from __future__ import annotations
import struct
import hashlib
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from functools import cached_property

# NKNN v2 constants
NKNN_MAGIC = 0x4E4B4E4E  # "NKNN" in ASCII
NKNN_VERSION = 2

# Architecture constants
INPUT_SIZE = 40960      # HalfKP features per perspective
L1_SIZE = 256           # First hidden layer
L2_SIZE = 32            # Second hidden layer
L3_SIZE = 32            # Third hidden layer
OUTPUT_SIZE = 1         # Eval output
WDL_SIZE = 3            # Win/Draw/Loss logits

# Quantization scales
QUANT_SCALE_W1 = 128.0  # Sparse layer weights/biases
QUANT_SCALE_DENSE = 64.0  # Dense layer weights
QUANT_SCALE_BIAS = 128.0  # Dense layer biases


@dataclass
class NKNNModel:
    """Loaded NKNN model with dequantized weights."""
    W1_q: np.ndarray  # [40960, 256] i16
    B1_q: np.ndarray  # [256] i16
    W2_q: np.ndarray  # [512, 32] i8
    B2_q: np.ndarray  # [32] i16
    W3_q: np.ndarray  # [32, 32] i8
    B3_q: np.ndarray  # [32] i16
    W4_q: np.ndarray  # [32, 1] i8
    B4_q: np.ndarray  # [1] i16
    W_wdl_q: np.ndarray  # [32, 3] i8
    B_wdl_q: np.ndarray  # [3] i16
    file_path: str
    file_size: int
    sha256: str

    @cached_property
    def W1(self) -> np.ndarray:
        return self.W1_q.astype(np.float32) / QUANT_SCALE_W1

    @cached_property
    def B1(self) -> np.ndarray:
        return self.B1_q.astype(np.float32) / QUANT_SCALE_W1

    @cached_property
    def W2(self) -> np.ndarray:
        return self.W2_q.astype(np.float32) / QUANT_SCALE_DENSE

    @cached_property
    def B2(self) -> np.ndarray:
        return self.B2_q.astype(np.float32) / QUANT_SCALE_BIAS

    @cached_property
    def W3(self) -> np.ndarray:
        return self.W3_q.astype(np.float32) / QUANT_SCALE_DENSE

    @cached_property
    def B3(self) -> np.ndarray:
        return self.B3_q.astype(np.float32) / QUANT_SCALE_BIAS

    @cached_property
    def W4(self) -> np.ndarray:
        return self.W4_q.astype(np.float32) / QUANT_SCALE_DENSE

    @cached_property
    def B4(self) -> np.ndarray:
        return self.B4_q.astype(np.float32) / QUANT_SCALE_BIAS

    @cached_property
    def W_wdl(self) -> np.ndarray:
        return self.W_wdl_q.astype(np.float32) / QUANT_SCALE_DENSE

    @cached_property
    def B_wdl(self) -> np.ndarray:
        return self.B_wdl_q.astype(np.float32) / QUANT_SCALE_BIAS


def load_nknn(path: Path) -> NKNNModel:
    """Load NKNN v2 model from binary file."""
    data = path.read_bytes()
    file_size = len(data)
    sha256 = hashlib.sha256(data).hexdigest().upper()

    magic, version = struct.unpack_from("<II", data, 0)
    if magic != NKNN_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:08X}")
    if version != NKNN_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    offset = 8

    w1_size = INPUT_SIZE * L1_SIZE
    W1_q = np.frombuffer(data, dtype="<i2", count=w1_size, offset=offset).reshape(INPUT_SIZE, L1_SIZE)
    offset += w1_size * 2

    B1_q = np.frombuffer(data, dtype="<i2", count=L1_SIZE, offset=offset)
    offset += L1_SIZE * 2

    w2_rows = L1_SIZE * 2
    W2_q = np.frombuffer(data, dtype="<i1", count=w2_rows * L2_SIZE, offset=offset).reshape(w2_rows, L2_SIZE)
    offset += w2_rows * L2_SIZE

    B2_q = np.frombuffer(data, dtype="<i2", count=L2_SIZE, offset=offset)
    offset += L2_SIZE * 2

    W3_q = np.frombuffer(data, dtype="<i1", count=L3_SIZE * L3_SIZE, offset=offset).reshape(L3_SIZE, L3_SIZE)
    offset += L3_SIZE * L3_SIZE

    B3_q = np.frombuffer(data, dtype="<i2", count=L3_SIZE, offset=offset)
    offset += L3_SIZE * 2

    W4_q = np.frombuffer(data, dtype="<i1", count=L3_SIZE * OUTPUT_SIZE, offset=offset).reshape(L3_SIZE, OUTPUT_SIZE)
    offset += L3_SIZE * OUTPUT_SIZE

    B4_q = np.frombuffer(data, dtype="<i2", count=OUTPUT_SIZE, offset=offset)
    offset += OUTPUT_SIZE * 2

    W_wdl_q = np.frombuffer(data, dtype="<i1", count=L3_SIZE * WDL_SIZE, offset=offset).reshape(L3_SIZE, WDL_SIZE)
    offset += L3_SIZE * WDL_SIZE

    B_wdl_q = np.frombuffer(data, dtype="<i2", count=WDL_SIZE, offset=offset)
    offset += WDL_SIZE * 2

    return NKNNModel(
        W1_q=W1_q.copy(), B1_q=B1_q.copy(),
        W2_q=W2_q.copy(), B2_q=B2_q.copy(),
        W3_q=W3_q.copy(), B3_q=B3_q.copy(),
        W4_q=W4_q.copy(), B4_q=B4_q.copy(),
        W_wdl_q=W_wdl_q.copy(), B_wdl_q=B_wdl_q.copy(),
        file_path=str(path),
        file_size=file_size,
        sha256=sha256
    )


def print_model_info(model: NKNNModel) -> None:
    print("=" * 60)
    print("NKNN v2 Model Info")
    print("=" * 60)
    print(f"File:     {model.file_path}")
    print(f"Size:     {model.file_size:,} bytes")
    print(f"SHA-256:  {model.sha256}")
    print()
    print("Architecture: 40960 -> 256 -> 32 -> 32 -> 1 (+WDL)")
    print()
    print("Layer Statistics (dequantized):")
    print("-" * 60)

    layers = [
        ("W1 (sparse)", model.W1, model.W1.shape),
        ("B1", model.B1, model.B1.shape),
        ("W2 (dense)", model.W2, model.W2.shape),
        ("B2", model.B2, model.B2.shape),
        ("W3 (dense)", model.W3, model.W3.shape),
        ("B3", model.B3, model.B3.shape),
        ("W4 (dense)", model.W4, model.W4.shape),
        ("B4", model.B4, model.B4.shape),
        ("W_wdl", model.W_wdl, model.W_wdl.shape),
        ("B_wdl", model.B_wdl, model.B_wdl.shape),
    ]

    for name, arr, shape in layers:
        print(f"{name:15} shape={str(shape):15} "
              f"min={arr.min():8.4f} max={arr.max():8.4f} "
              f"mean={arr.mean():8.4f} std={arr.std():8.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect NKNN v2 model files")
    ap.add_argument("--model", "-m", required=True, help="Path to .nknn file")
    ap.add_argument("--quiet", "-q", action="store_true", help="Only print hash")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return 1

    try:
        model = load_nknn(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    if args.quiet:
        print(f"sha256={model.sha256}")
    else:
        print_model_info(model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
