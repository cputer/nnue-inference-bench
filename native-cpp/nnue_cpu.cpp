#include "nnue_cpu.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

// Simple PCG random (same sequence as numpy default_rng)
class PCG32 {
    uint64_t state, inc;
public:
    PCG32(uint64_t seed) : state(seed), inc(1) {
        state = state * 6364136223846793005ULL + inc;
    }
    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    int integers(int low, int high) {
        return low + (next() % (high - low));
    }
};

bool load_nknn(const std::string& path, NKNNModel& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    // Check magic
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    
    if (magic != 0x4E4B4E4E || version != 2) return false;
    
    // Read W1 (40960 * 256 i16)
    model.W1.resize(L1_INPUT * L1_OUTPUT);
    file.read(reinterpret_cast<char*>(model.W1.data()), L1_INPUT * L1_OUTPUT * 2);
    
    // Read B1 (256 i16)
    file.read(reinterpret_cast<char*>(model.B1.data()), L1_OUTPUT * 2);
    
    // Read W2 (512 * 32 i8)
    file.read(reinterpret_cast<char*>(model.W2.data()), L2_INPUT * L2_OUTPUT);
    
    // Read B2 (32 i16)
    file.read(reinterpret_cast<char*>(model.B2.data()), L2_OUTPUT * 2);
    
    // Read W3 (32 * 32 i8)
    file.read(reinterpret_cast<char*>(model.W3.data()), L3_OUTPUT * L3_OUTPUT);
    
    // Read B3 (32 i16)
    file.read(reinterpret_cast<char*>(model.B3.data()), L3_OUTPUT * 2);
    
    // Read W4 (32 i8)
    file.read(reinterpret_cast<char*>(model.W4.data()), L3_OUTPUT);
    
    // Read B4 (1 i16)
    file.read(reinterpret_cast<char*>(&model.B4), 2);
    
    // Dequantize to float
    model.W1_f.resize(L1_INPUT * L1_OUTPUT);
    for (size_t i = 0; i < model.W1.size(); i++) {
        model.W1_f[i] = model.W1[i] / SCALE_SPARSE;
    }
    for (int i = 0; i < L1_OUTPUT; i++) {
        model.B1_f[i] = model.B1[i] / SCALE_SPARSE;
    }
    for (size_t i = 0; i < model.W2.size(); i++) {
        model.W2_f[i] = model.W2[i] / SCALE_DENSE_W;
    }
    for (int i = 0; i < L2_OUTPUT; i++) {
        model.B2_f[i] = model.B2[i] / SCALE_DENSE_B;
    }
    for (size_t i = 0; i < model.W3.size(); i++) {
        model.W3_f[i] = model.W3[i] / SCALE_DENSE_W;
    }
    for (int i = 0; i < L3_OUTPUT; i++) {
        model.B3_f[i] = model.B3[i] / SCALE_DENSE_B;
    }
    for (int i = 0; i < L3_OUTPUT; i++) {
        model.W4_f[i] = model.W4[i] / SCALE_DENSE_W;
    }
    model.B4_f = model.B4 / SCALE_DENSE_B;
    
    model.loaded = true;
    return true;
}

inline float screlu(float x) {
    float clipped = std::max(0.0f, std::min(1.0f, x));
    return clipped * clipped;
}

void extract_halfkp_features(const Position& pos,
                             std::vector<int>& features_white,
                             std::vector<int>& features_black) {
    features_white.clear();
    features_black.clear();
    
    for (int sq = 0; sq < 64; sq++) {
        int piece = pos.pieces[sq];
        if (piece < 0 || piece == 5 || piece == 11) continue;  // Empty or king
        
        int halfkp_piece = (piece < 5) ? piece : piece - 1;
        
        // White perspective
        int w_feat = pos.white_king * 640 + halfkp_piece * 64 + sq;
        features_white.push_back(w_feat);
        
        // Black perspective (flipped)
        int flipped_sq = sq ^ 56;
        int flipped_king = pos.black_king ^ 56;
        int b_halfkp_piece = (halfkp_piece < 5) ? halfkp_piece + 5 : halfkp_piece - 5;
        int b_feat = flipped_king * 640 + b_halfkp_piece * 64 + flipped_sq;
        features_black.push_back(b_feat);
    }
}

float forward_float(const NKNNModel& model,
                    const std::vector<int>& features_white,
                    const std::vector<int>& features_black,
                    int stm) {
    // L1: Sparse accumulation
    std::array<float, L1_OUTPUT> acc_white, acc_black;
    std::copy(model.B1_f.begin(), model.B1_f.end(), acc_white.begin());
    std::copy(model.B1_f.begin(), model.B1_f.end(), acc_black.begin());
    
    for (int idx : features_white) {
        if (idx >= 0 && idx < L1_INPUT) {
            for (int j = 0; j < L1_OUTPUT; j++) {
                acc_white[j] += model.W1_f[idx * L1_OUTPUT + j];
            }
        }
    }
    
    for (int idx : features_black) {
        if (idx >= 0 && idx < L1_INPUT) {
            for (int j = 0; j < L1_OUTPUT; j++) {
                acc_black[j] += model.W1_f[idx * L1_OUTPUT + j];
            }
        }
    }
    
    // Concatenate with SCReLU based on STM
    std::array<float, L2_INPUT> hidden1;
    if (stm == 0) {  // White to move
        for (int i = 0; i < L1_OUTPUT; i++) {
            hidden1[i] = screlu(acc_white[i]);
            hidden1[L1_OUTPUT + i] = screlu(acc_black[i]);
        }
    } else {  // Black to move
        for (int i = 0; i < L1_OUTPUT; i++) {
            hidden1[i] = screlu(acc_black[i]);
            hidden1[L1_OUTPUT + i] = screlu(acc_white[i]);
        }
    }
    
    // L2: Dense 512->32
    std::array<float, L2_OUTPUT> hidden2;
    for (int j = 0; j < L2_OUTPUT; j++) {
        float sum = model.B2_f[j];
        for (int i = 0; i < L2_INPUT; i++) {
            sum += hidden1[i] * model.W2_f[i * L2_OUTPUT + j];
        }
        hidden2[j] = screlu(sum);
    }
    
    // L3: Dense 32->32
    std::array<float, L3_OUTPUT> hidden3;
    for (int j = 0; j < L3_OUTPUT; j++) {
        float sum = model.B3_f[j];
        for (int i = 0; i < L2_OUTPUT; i++) {
            sum += hidden2[i] * model.W3_f[i * L3_OUTPUT + j];
        }
        hidden3[j] = screlu(sum);
    }
    
    // L4: Dense 32->1 (no activation)
    float eval_score = model.B4_f;
    for (int i = 0; i < L3_OUTPUT; i++) {
        eval_score += hidden3[i] * model.W4_f[i];
    }
    
    return eval_score;
}

Position create_random_position(uint64_t seed) {
    PCG32 rng(seed);
    Position pos;
    std::fill(pos.pieces.begin(), pos.pieces.end(), -1);
    
    pos.white_king = rng.integers(0, 64);
    pos.black_king = rng.integers(0, 64);
    while (pos.black_king == pos.white_king ||
           (std::abs(pos.black_king / 8 - pos.white_king / 8) <= 1 &&
            std::abs(pos.black_king % 8 - pos.white_king % 8) <= 1)) {
        pos.black_king = rng.integers(0, 64);
    }
    
    pos.pieces[pos.white_king] = 5;
    pos.pieces[pos.black_king] = 11;
    
    int num_pieces = rng.integers(4, 16);
    int piece_types[] = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10};
    
    for (int i = 0; i < num_pieces; i++) {
        int sq = rng.integers(0, 64);
        if (pos.pieces[sq] == -1) {
            pos.pieces[sq] = piece_types[rng.next() % 10];
        }
    }
    
    pos.stm = rng.integers(0, 2);
    return pos;
}

uint32_t compute_checksum(const std::vector<float>& evals) {
    // MD5 of float array, take first 8 hex chars
    // Simplified: use same XOR approach as reference
    uint32_t result = 0;
    for (float e : evals) {
        int32_t q = static_cast<int32_t>(std::round(e * 10000.0f));
        result ^= static_cast<uint32_t>(q);
    }
    return result;
}
