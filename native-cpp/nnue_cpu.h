#ifndef NNUE_CPU_H
#define NNUE_CPU_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>

// NKNN v2 model constants
constexpr int L1_INPUT = 40960;
constexpr int L1_OUTPUT = 256;
constexpr int L2_INPUT = 512;
constexpr int L2_OUTPUT = 32;
constexpr int L3_OUTPUT = 32;
constexpr int L4_OUTPUT = 1;

// Quantization scales
constexpr float SCALE_SPARSE = 128.0f;
constexpr float SCALE_DENSE_W = 64.0f;
constexpr float SCALE_DENSE_B = 128.0f;

struct NKNNModel {
    // Sparse layer (i16)
    std::vector<int16_t> W1;  // [40960 * 256]
    std::array<int16_t, L1_OUTPUT> B1;
    
    // Dense layers (i8 weights, i16 biases)
    std::array<int8_t, L2_INPUT * L2_OUTPUT> W2;
    std::array<int16_t, L2_OUTPUT> B2;
    
    std::array<int8_t, L3_OUTPUT * L3_OUTPUT> W3;
    std::array<int16_t, L3_OUTPUT> B3;
    
    std::array<int8_t, L3_OUTPUT> W4;
    int16_t B4;
    
    // Dequantized float weights for CPU inference
    std::vector<float> W1_f;  // [40960 * 256]
    std::array<float, L1_OUTPUT> B1_f;
    std::array<float, L2_INPUT * L2_OUTPUT> W2_f;
    std::array<float, L2_OUTPUT> B2_f;
    std::array<float, L3_OUTPUT * L3_OUTPUT> W3_f;
    std::array<float, L3_OUTPUT> B3_f;
    std::array<float, L3_OUTPUT> W4_f;
    float B4_f;
    
    bool loaded = false;
};

struct Position {
    std::array<int8_t, 64> pieces;  // -1 = empty, 0-4 = WP..WQ, 5=WK, 6-10=BP..BQ, 11=BK
    int white_king;
    int black_king;
    int stm;  // 0 = white, 1 = black
};

// Load model from file
bool load_nknn(const std::string& path, NKNNModel& model);

// Extract HalfKP features
void extract_halfkp_features(const Position& pos,
                             std::vector<int>& features_white,
                             std::vector<int>& features_black);

// Forward pass (float)
float forward_float(const NKNNModel& model,
                    const std::vector<int>& features_white,
                    const std::vector<int>& features_black,
                    int stm);

// Generate random position
Position create_random_position(uint64_t seed);

// Compute checksum (same as Python)
uint32_t compute_checksum(const std::vector<float>& evals);

#endif // NNUE_CPU_H
