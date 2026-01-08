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
    base = Path(__file__).parent.parent / "native-cuda"
    dll_candidates = [
        base / "build" / "nnue_cuda.dll",
        base / "nnue_cuda.dll",
        base / "build" / "nnue_cuda.so",
        base / "nnue_cuda.so",
    ]
    kernel_path = None
    for p in dll_candidates:
        if p.exists():
            kernel_path = p
            break

    if kernel_path is None:
        status["blocked_reason"] = "CUDA kernels not compiled (run build.ps1)"
        return status

    # Try to load the library
    try:
        lib = ctypes.CDLL(str(kernel_path))
        _setup_lib_signatures(lib)
        err = lib.nnue_init(0)
        if err != 0:
            status["blocked_reason"] = f"CUDA init failed (error {err})"
            return status
        status["available"] = True
        status["_lib"] = lib
    except OSError as e:
        status["blocked_reason"] = f"Failed to load CUDA kernels: {e}"
        return status

    return status


def _setup_lib_signatures(lib):
    """Set up ctypes function signatures for the CUDA library."""
    lib.nnue_get_version.argtypes = []
    lib.nnue_get_version.restype = ctypes.c_int
    lib.nnue_init.argtypes = [ctypes.c_int]
    lib.nnue_init.restype = ctypes.c_int
    lib.nnue_shutdown.argtypes = []
    lib.nnue_shutdown.restype = ctypes.c_int
    lib.nnue_sync.argtypes = []
    lib.nnue_sync.restype = ctypes.c_int
    lib.nnue_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.nnue_malloc.restype = ctypes.c_int
    lib.nnue_free.argtypes = [ctypes.c_void_p]
    lib.nnue_free.restype = ctypes.c_int
    lib.nnue_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.nnue_memcpy_h2d.restype = ctypes.c_int
    lib.nnue_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.nnue_memcpy_d2h.restype = ctypes.c_int
    lib.nnue_sparse_accum.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    lib.nnue_sparse_accum.restype = ctypes.c_int
    lib.nnue_dense_layer.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    lib.nnue_dense_layer.restype = ctypes.c_int
    lib.nnue_concat_accum.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    lib.nnue_concat_accum.restype = ctypes.c_int


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


L1_SIZE = 256
MAX_FEATURES = 64


class GPUInference:
    """GPU inference using CUDA kernels."""

    def __init__(self, model):
        """Initialize GPU inference with model weights."""
        status = get_gpu_status()
        if not status["available"]:
            raise RuntimeError(f"GPU not available: {status['blocked_reason']}")
        self.model = model
        self.lib = _GPU_STATUS.get("_lib")
        self._alloc_weights()
        self._upload_weights()

    def _alloc_weights(self):
        """Allocate GPU memory for weights."""
        self.d_W1 = ctypes.c_void_p()
        self.d_B1 = ctypes.c_void_p()
        self.d_W2 = ctypes.c_void_p()
        self.d_B2 = ctypes.c_void_p()
        self.d_W3 = ctypes.c_void_p()
        self.d_B3 = ctypes.c_void_p()
        self.d_W4 = ctypes.c_void_p()
        self.d_B4 = ctypes.c_void_p()
        self.lib.nnue_malloc(ctypes.byref(self.d_W1), 40960 * L1_SIZE * 2)
        self.lib.nnue_malloc(ctypes.byref(self.d_B1), L1_SIZE * 2)
        self.lib.nnue_malloc(ctypes.byref(self.d_W2), 512 * 32)
        self.lib.nnue_malloc(ctypes.byref(self.d_B2), 32 * 2)
        self.lib.nnue_malloc(ctypes.byref(self.d_W3), 32 * 32)
        self.lib.nnue_malloc(ctypes.byref(self.d_B3), 32 * 2)
        self.lib.nnue_malloc(ctypes.byref(self.d_W4), 32)
        self.lib.nnue_malloc(ctypes.byref(self.d_B4), 2)

    def _upload_weights(self):
        """Upload model weights to GPU."""
        m = self.model
        for nm, dp, dt in [("W1_q", self.d_W1, np.int16), ("B1_q", self.d_B1, np.int16),
                           ("W2_q", self.d_W2, np.int8), ("B2_q", self.d_B2, np.int16),
                           ("W3_q", self.d_W3, np.int8), ("B3_q", self.d_B3, np.int16),
                           ("W4_q", self.d_W4, np.int8), ("B4_q", self.d_B4, np.int16)]:
            a = getattr(m, nm).astype(dt)
            self.lib.nnue_memcpy_h2d(dp, a.ctypes.data, a.nbytes)
        self.lib.nnue_sync()

    def forward_batch(self, features_white, features_black, stm) -> np.ndarray:
        """Run GPU inference on a batch of positions."""
        bs = len(stm)
        fw = np.zeros((bs, MAX_FEATURES), np.int32)
        fb = np.zeros((bs, MAX_FEATURES), np.int32)
        cw = np.zeros(bs, np.int32)
        cb = np.zeros(bs, np.int32)
        for i in range(bs):
            cw[i] = min(len(features_white[i]), MAX_FEATURES)
            cb[i] = min(len(features_black[i]), MAX_FEATURES)
            fw[i, :cw[i]] = features_white[i][:cw[i]]
            fb[i, :cb[i]] = features_black[i][:cb[i]]
        stm_arr = np.array(stm, np.int32)

        # Allocate batch buffers
        bufs = {}
        for nm, sz in [("fw", fw.nbytes), ("fb", fb.nbytes), ("cw", cw.nbytes), ("cb", cb.nbytes),
                       ("stm", stm_arr.nbytes), ("aw", bs * 256 * 4), ("ab", bs * 256 * 4),
                       ("cat", bs * 512 * 4), ("l2", bs * 32 * 4), ("l3", bs * 32 * 4), ("l4", bs * 4)]:
            bufs[nm] = ctypes.c_void_p()
            self.lib.nnue_malloc(ctypes.byref(bufs[nm]), sz)

        # Upload input
        self.lib.nnue_memcpy_h2d(bufs["fw"], fw.ctypes.data, fw.nbytes)
        self.lib.nnue_memcpy_h2d(bufs["fb"], fb.ctypes.data, fb.nbytes)
        self.lib.nnue_memcpy_h2d(bufs["cw"], cw.ctypes.data, cw.nbytes)
        self.lib.nnue_memcpy_h2d(bufs["cb"], cb.ctypes.data, cb.nbytes)
        self.lib.nnue_memcpy_h2d(bufs["stm"], stm_arr.ctypes.data, stm_arr.nbytes)

        # Sparse accum
        self.lib.nnue_sparse_accum(self.d_W1, self.d_B1, bufs["fw"], bufs["cw"], bufs["aw"], bs, MAX_FEATURES)
        self.lib.nnue_sparse_accum(self.d_W1, self.d_B1, bufs["fb"], bufs["cb"], bufs["ab"], bs, MAX_FEATURES)
        self.lib.nnue_concat_accum(bufs["aw"], bufs["ab"], bufs["stm"], bufs["cat"], bs)

        # Dense layers
        self.lib.nnue_dense_layer(self.d_W2, self.d_B2, bufs["cat"], bufs["l2"], bs, 512, 32, 1)
        self.lib.nnue_dense_layer(self.d_W3, self.d_B3, bufs["l2"], bufs["l3"], bs, 32, 32, 1)
        self.lib.nnue_dense_layer(self.d_W4, self.d_B4, bufs["l3"], bufs["l4"], bs, 32, 1, 0)

        # Download
        out = np.zeros(bs, np.float32)
        self.lib.nnue_memcpy_d2h(out.ctypes.data, bufs["l4"], out.nbytes)
        self.lib.nnue_sync()

        # Free batch buffers
        for ptr in bufs.values():
            self.lib.nnue_free(ptr)
        return out

    def __del__(self):
        if hasattr(self, "lib") and self.lib:
            for a in ["d_W1", "d_B1", "d_W2", "d_B2", "d_W3", "d_B3", "d_W4", "d_B4"]:
                ptr = getattr(self, a, None)
                if ptr and ptr.value:
                    self.lib.nnue_free(ptr)


def forward_gpu_batch(model, positions):
    """GPU batch forward pass."""
    import time
    if not is_gpu_available():
        raise RuntimeError(f"GPU not available: {get_gpu_status()['blocked_reason']}")
    
    gpu = GPUInference(model)
    from infer_cpu import extract_halfkp_features
    
    features_w, features_b, stm_list = [], [], []
    for pos in positions:
        fw, fb = extract_halfkp_features(pos)
        features_w.append(fw)
        features_b.append(fb)
        stm_list.append(pos.stm)
    
    start = time.perf_counter()
    evals = gpu.forward_batch(features_w, features_b, stm_list)
    _GPU_STATUS["_lib"].nnue_sync()
    kernel_time = (time.perf_counter() - start) * 1000
    return evals, kernel_time


if __name__ == "__main__":
    status = get_gpu_status()
    print("GPU Status:")
    for k, v in status.items():
        print(f"  {k}: {v}")
    if status["available"]:
        print(f"CUDA API version: {_GPU_STATUS['_lib'].nnue_get_version()}")