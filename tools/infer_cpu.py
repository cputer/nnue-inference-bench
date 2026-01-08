#!/usr/bin/env python3
"""
CPU Reference Inference for NKNN v2.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inspect_nknn import NKNNModel, load_nknn

FEATURES_PER_PERSPECTIVE = 40960


@dataclass
class Position:
    pieces: np.ndarray
    white_king: int
    black_king: int
    stm: int


def screlu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped


def extract_halfkp_features(pos: Position) -> Tuple[List[int], List[int]]:
    white_features = []
    black_features = []

    for sq in range(64):
        piece = int(pos.pieces[sq])
        if piece < 0 or piece == 5 or piece == 11:
            continue

        if piece < 5:
            halfkp_piece = piece
        else:
            halfkp_piece = piece - 1

        w_feat = pos.white_king * 640 + halfkp_piece * 64 + sq
        white_features.append(w_feat)

        flipped_sq = sq ^ 56
        flipped_king = pos.black_king ^ 56
        if halfkp_piece < 5:
            b_halfkp_piece = halfkp_piece + 5
        else:
            b_halfkp_piece = halfkp_piece - 5

        b_feat = flipped_king * 640 + b_halfkp_piece * 64 + flipped_sq
        black_features.append(b_feat)

    return white_features, black_features


def forward_float(model: NKNNModel,
                  features_white: List[int],
                  features_black: List[int],
                  stm: int) -> Tuple[float, np.ndarray]:
    # Vectorized sparse accumulation (much faster than loop)
    acc_white = model.B1.copy()
    if features_white:
        valid_w = np.array([i for i in features_white if 0 <= i < 40960], dtype=np.int32)
        if len(valid_w) > 0:
            acc_white += model.W1[valid_w].sum(axis=0)

    acc_black = model.B1.copy()
    if features_black:
        valid_b = np.array([i for i in features_black if 0 <= i < 40960], dtype=np.int32)
        if len(valid_b) > 0:
            acc_black += model.W1[valid_b].sum(axis=0)

    if stm == 0:
        hidden = np.concatenate([screlu(acc_white), screlu(acc_black)])
    else:
        hidden = np.concatenate([screlu(acc_black), screlu(acc_white)])

    hidden = screlu(model.W2.T @ hidden + model.B2)
    hidden = screlu(model.W3.T @ hidden + model.B3)
    eval_score = float((model.W4.T @ hidden + model.B4)[0])
    wdl_logits = model.W_wdl.T @ hidden + model.B_wdl

    return eval_score, wdl_logits


def compute_checksum(evals: List[float]) -> int:
    import hashlib
    data = np.array(evals, dtype=np.float32).tobytes()
    return int(hashlib.md5(data).hexdigest()[:8], 16)


def create_random_position(seed: int) -> Position:
    rng = np.random.default_rng(seed)
    pieces = np.full(64, -1, dtype=np.int8)

    white_king = int(rng.integers(0, 64))
    black_king = int(rng.integers(0, 64))
    while black_king == white_king or (abs(black_king // 8 - white_king // 8) <= 1 and abs(black_king % 8 - white_king % 8) <= 1):
        black_king = int(rng.integers(0, 64))

    pieces[white_king] = 5
    pieces[black_king] = 11

    num_pieces = int(rng.integers(4, 16))
    piece_types = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

    for _ in range(num_pieces):
        sq = int(rng.integers(0, 64))
        if pieces[sq] == -1:
            pieces[sq] = int(rng.choice(piece_types))

    stm = int(rng.integers(0, 2))
    return Position(pieces=pieces, white_king=white_king, black_king=black_king, stm=stm)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CPU inference test")
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--positions", "-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    model = load_nknn(Path(args.model))
    print(f"SHA-256: {model.sha256}")
    print()

    print(f"Running inference on {args.positions} random positions...")
    evals = []
    for i in range(args.positions):
        pos = create_random_position(args.seed + i)
        feat_w, feat_b = extract_halfkp_features(pos)
        eval_score, wdl = forward_float(model, feat_w, feat_b, pos.stm)
        evals.append(eval_score)

        if i < 5:
            print(f"  pos {i}: features=({len(feat_w)},{len(feat_b)}) eval={eval_score:.4f} wdl={wdl}")

    checksum = compute_checksum(evals)
    print()
    print(f"Checksum (first {args.positions} positions): 0x{checksum:08X}")
    print(f"Mean eval: {np.mean(evals):.4f}")
    print(f"Std eval: {np.std(evals):.4f}")