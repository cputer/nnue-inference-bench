# NKNN v2 Binary Format Specification

This document defines the exact binary layout for the NNUE model format used by NikolaChess.
It is the **single source of truth** for all loaders, exporters, and validators.

## Overview

| Property | Value |
|----------|-------|
| Magic | `0x4E4B4E4E` ("NKNN" in ASCII) |
| Version | 2 |
| Endianness | Little-endian |
| Alignment | Tightly packed (no padding) |
| Total Size | 20,989,768 bytes |

## Network Architecture

```
Input: HalfKP features (40960 per perspective)
       |
+---------------------------------------------+
| L1 (Sparse): 40960 -> 256 (per side)        |  W1: i16, B1: i16
| Activation: SCReLU (Squared Clipped ReLU)   |
+---------------------------------------------+
       | concatenate white + black (512)
+---------------------------------------------+
| L2 (Dense): 512 -> 32                       |  W2: i8, B2: i16
| Activation: SCReLU                          |
+---------------------------------------------+
       |
+---------------------------------------------+
| L3 (Dense): 32 -> 32                        |  W3: i8, B3: i16
| Activation: SCReLU                          |
+---------------------------------------------+
       |
+---------------------------------------------+
| L4 (Dense): 32 -> 1                         |  W4: i8, B4: i16
| Output: raw eval score (centipawns)         |
+---------------------------------------------+
       |
+---------------------------------------------+
| WDL Head (Dense): 32 -> 3                   |  W_wdl: i8, B_wdl: i16
| Output: [win, draw, loss] logits            |
+---------------------------------------------+
```

## HalfKP Feature Encoding

Each position is encoded as sparse active features for white and black perspectives:

```
feature_index = king_square * 640 + piece_type * 64 + piece_square

where:
  king_square  in [0, 63]   (own king location)
  piece_type   in [0, 9]    (WP,WN,WB,WR,WQ, BP,BN,BB,BR,BQ)
  piece_square in [0, 63]   (piece location)

Total features per perspective: 64 * 10 * 64 = 40960
```

Typical active features per side: 4-30 (sparse)

## Quantization Scheme

| Tensor | Data Type | Scale Factor | Notes |
|--------|-----------|--------------|-------|
| W1 (sparse layer weights) | i16 | 128.0 | quantized = round(float * 128) |
| B1 (sparse layer bias) | i16 | 128.0 | |
| W2, W3, W4 (dense weights) | i8 | 64.0 | quantized = round(float * 64) |
| B2, B3, B4 (dense biases) | i16 | 128.0 | |
| W_wdl (WDL head weights) | i8 | 64.0 | |
| B_wdl (WDL head bias) | i16 | 128.0 | |

**Dequantization**: float_value = quantized_value / scale

## Binary Layout

All multi-byte integers are **little-endian**.

```
Offset      Size (bytes)    Field
--------------------------------------------------------------
0x00000000  4               magic (0x4E4B4E4E)
0x00000004  4               version (2)

# Layer 1: Sparse (40960 -> 256), weights stored row-major
0x00000008  20,971,520      W1[40960][256] as i16 (40960 * 256 * 2)
0x01400008  512             B1[256] as i16 (256 * 2)

# Layer 2: Dense (512 -> 32)
0x01400208  16,384          W2[512][32] as i8 (512 * 32 * 1)
0x01404208  64              B2[32] as i16 (32 * 2)

# Layer 3: Dense (32 -> 32)
0x01404248  1,024           W3[32][32] as i8 (32 * 32 * 1)
0x01404648  64              B3[32] as i16 (32 * 2)

# Layer 4: Dense (32 -> 1) - eval head
0x01404688  32              W4[32][1] as i8 (32 * 1 * 1)
0x014046A8  2               B4[1] as i16 (1 * 2)

# WDL Head: Dense (32 -> 3)
0x014046AA  96              W_wdl[32][3] as i8 (32 * 3 * 1)
0x0140470A  6               B_wdl[3] as i16 (3 * 2)

--------------------------------------------------------------
Total:      20,989,768 bytes (0x01404710)
```

## Size Calculation

```
Header:           8
W1:   40960 * 256 * 2 = 20,971,520
B1:     256 * 2       =        512
W2:     512 * 32 * 1  =     16,384
B2:      32 * 2       =         64
W3:      32 * 32 * 1  =      1,024
B3:      32 * 2       =         64
W4:      32 * 1 * 1   =         32
B4:       1 * 2       =          2
W_wdl:   32 * 3 * 1   =         96
B_wdl:    3 * 2       =          6
---------------------------------
Total:                = 20,989,708 bytes + 8 header = 20,989,716 bytes
```

Note: Actual file size may include alignment padding at end.

## SCReLU Activation

Squared Clipped ReLU used throughout:

```python
def screlu(x):
    clipped = max(0.0, min(1.0, x))  # Clamp to [0, 1]
    return clipped * clipped          # Square
```

In quantized inference, input is scaled appropriately before clamping.

## Inference Algorithm (Pseudocode)

```python
def forward(features_white, features_black, stm):
    """
    features_white: list of active feature indices for white perspective
    features_black: list of active feature indices for black perspective
    stm: side to move (0=white, 1=black)
    Returns: evaluation in centipawns from side-to-move perspective
    """
    # L1: Sparse accumulation (start from bias, add active feature weights)
    acc_white = B1.copy()  # [256] i16
    for idx in features_white:
        acc_white += W1[idx]  # Add row idx of W1

    acc_black = B1.copy()
    for idx in features_black:
        acc_black += W1[idx]

    # Apply SCReLU and concatenate based on side to move
    if stm == 0:  # White to move
        hidden = concat(screlu(acc_white), screlu(acc_black))  # [512]
    else:         # Black to move
        hidden = concat(screlu(acc_black), screlu(acc_white))  # [512]

    # L2: Dense 512->32 with SCReLU
    hidden = screlu(W2 @ hidden + B2)  # [32]

    # L3: Dense 32->32 with SCReLU
    hidden = screlu(W3 @ hidden + B3)  # [32]

    # L4: Dense 32->1 (no activation, raw score)
    eval_score = W4 @ hidden + B4      # scalar

    return eval_score
```

## Validation

To verify a model file:

1. Check magic bytes = 4E 4B 4E 4E (little-endian: 0x4E4B4E4E)
2. Check version = 2
3. Check file size matches expected
4. Compute SHA-256 hash for provenance tracking

## Reference Implementation

Gold model for testing: models/nikola_d12v2_gold.nknn
- SHA-256: 041BC71A527B830E54A83664BD7B2C4414A9F047348943DF2CE3359B49142E3C
- Trained on depth-12 data with fixed sparse optimizer

## Compatibility Notes

- This format is specific to NikolaChess NNUE architecture
- Not compatible with Stockfish NNUE (.nnue) files
- Version 1 used different quantization scales (deprecated)
