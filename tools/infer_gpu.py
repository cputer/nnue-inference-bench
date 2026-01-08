#!/usr/bin/env python3
"""
GPU Inference for NKNN v2 using CUDA kernels.

Loads native CUDA kernels via ctypes if available.
Falls back gracefully with documented reason if GPU unavailable.
"""

from __future__ import annotations
import os
import sys
import ctypes
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# GPU availability status
_GPU_STATUS: Dict[str, Any] = {
    "available": False,
    "blocked_reason": None,
    "gpu_name": None,
    "cuda_version": None,
}


def _detect_gpu() -> Dict[str, Any]:
    """Detect GPU availability and return status dict."""
    status = {
        "available": False,
        "blocked_reason": None,
        "gpu_name": None,
        "cuda_version": None,
    }

    # Check for NVIDIA GPU via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            status["gpu_name"] = parts[0].strip() if parts else "Unknown"
            status["cuda_version"] = parts[1].strip() if len(parts) > 1 else "Unknown"
        else:
            status["blocked_reason"] = "nvidia-smi failed"
            return status
    except FileNotFoundError:
        status["blocked_reason"] = "nvidia-smi not found"
        return status
    except Exception as e:
        status["blocked_reason"] = f"GPU detection error: {e}"
        return status

    # Check for compiled CUDA kernels
    kernel_path = Path(__file__).parent.parent / "native-cuda" / "nnue_kernels.dll"
    if not kernel_path.exists():
        kernel_path = Path(__file__).parent.parent / "native-cuda" / "nnue_kernels.so"

    if not kernel_path.exists():
        status["blocked_reason"] = "CUDA kernels not compiled (run build.ps1 with CUDA toolkit)"
        return status

    # Try to load the library
    try:
        lib = ctypes.CDLL(str(kernel_path))
        status["available"] = True
        status["_lib"] = lib
    except OSError as e:
        status["blocked_reason"] = f"Failed to load CUDA kernels: {e}"
        return status

    return status


def get_gpu_status() -> Dict[str, Any]:
    """Get GPU availability status."""
    global _GPU_STATUS
    if _GPU_STATUS["blocked_reason"] is None and not _GPU_STATUS["available"]:
        _GPU_STATUS = _detect_gpu()
    return {k: v for k, v in _GPU_STATUS.items() if not k.startswith("_")}


def is_gpu_available() -> bool:
    """Check if GPU inference is available."""
    return get_gpu_status()["available"]


@dataclass
class GPUContext:
    """GPU context holding device memory and kernels."""
    lib: Any  # ctypes library
    # Device memory pointers would go here
    # For now, using numpy arrays with .ctypes interface


class GPUInference:
    """GPU inference wrapper."""

    def __init__(self, model):
        """Initialize GPU inference with model."""
        status = get_gpu_status()
        if not status["available"]:
            raise RuntimeError(f"GPU not available: {status['blocked_reason']}")

        self.model = model
        self.lib = _GPU_STATUS.get("_lib")
        self._setup_kernels()

    def _setup_kernels(self):
        """Set up kernel function signatures."""
        # nnue_sparse_accum
        self.lib.nnue_sparse_accum.argtypes = [
            ctypes.POINTER(ctypes.c_int16),  # W1
            ctypes.POINTER(ctypes.c_int16),  # B1
            ctypes.POINTER(ctypes.c_int32),  # features
            ctypes.POINTER(ctypes.c_int32),  # feature_counts
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # max_features
        ]
        self.lib.nnue_sparse_accum.restype = ctypes.c_int

        # nnue_dense_layer
        self.lib.nnue_dense_layer.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # weights
            ctypes.POINTER(ctypes.c_int16),  # bias
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # in_size
            ctypes.c_int,  # out_size
            ctypes.c_int,  # apply_screlu
        ]
        self.lib.nnue_dense_layer.restype = ctypes.c_int

    def forward_batch(
        self,
        features_white: List[List[int]],
        features_black: List[List[int]],
        stm: List[int]
    ) -> np.ndarray:
        """
        Run GPU inference on a batch of positions.

        Returns array of eval scores.
        """
        # This would use GPU kernels
        # For now, fallback to CPU
        raise NotImplementedError("GPU inference requires compiled CUDA kernels")


def forward_gpu(model, features_white, features_black, stm) -> Tuple[float, np.ndarray]:
    """
    GPU forward pass (single position).

    Falls back to CPU if GPU unavailable.
    """
    if not is_gpu_available():
        # Import CPU inference and use that
        from infer_cpu import forward_float
        return forward_float(model, features_white, features_black, stm)

    # GPU path would go here
    raise NotImplementedError("GPU single inference not yet implemented")


if __name__ == "__main__":
    status = get_gpu_status()
    print("GPU Status:")
    for k, v in status.items():
        print(f"  {k}: {v}")