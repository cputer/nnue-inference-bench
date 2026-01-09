// nnue_cpu_simd.cpp - SIMD-optimized NNUE inference (optimized SIMD)
// This shows AVX2 SIMD optimization

#include "nnue_cpu.h"
#include <immintrin.h>  // AVX2
#include <cstring>
#include <cmath>
#include <algorithm>

// Scalar SCReLU
inline float screlu(float x) {
    float clipped = std::max(0.0f, std::min(1.0f, x));
    return clipped * clipped;
}

// SIMD-optimized sparse accumulation using AVX2
void sparse_accum_simd(const NKNNModel& model,
                       const std::vector<int>& features,
                       std::array<float, L1_OUTPUT>& out) {
    // Initialize with bias using SIMD
    for (int i = 0; i < L1_OUTPUT; i += 8) {
        __m256 bias = _mm256_loadu_ps(&model.B1_f[i]);
        _mm256_storeu_ps(&out[i], bias);
    }
    
    // Accumulate feature weights (hot loop)
    for (int idx : features) {
        if (idx < 0 || idx >= L1_INPUT) continue;
        const float* row = &model.W1_f[idx * L1_OUTPUT];
        
        for (int i = 0; i < L1_OUTPUT; i += 8) {
            __m256 acc = _mm256_loadu_ps(&out[i]);
            __m256 weights = _mm256_loadu_ps(&row[i]);
            acc = _mm256_add_ps(acc, weights);
            _mm256_storeu_ps(&out[i], acc);
        }
    }
}

// SIMD SCReLU
inline __m256 screlu_avx(__m256 x) {
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clipped = _mm256_max_ps(zero, _mm256_min_ps(one, x));
    return _mm256_mul_ps(clipped, clipped);
}

// SIMD-optimized dense layer
void dense_layer_simd(int in_size, int out_size,
                      const float* weights,
                      const float* bias,
                      const float* input,
                      float* output,
                      bool apply_screlu) {
    for (int j = 0; j < out_size; j++) {
        __m256 sum = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        int i = 0;
        for (; i + 8 <= in_size; i += 8) {
            __m256 w = _mm256_loadu_ps(&weights[j * in_size + i]);
            __m256 in = _mm256_loadu_ps(&input[i]);
            sum = _mm256_fmadd_ps(w, in, sum);
        }
        
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128) + bias[j];
        
        // Handle remainder
        for (; i < in_size; i++) {
            result += weights[j * in_size + i] * input[i];
        }
        
        output[j] = apply_screlu ? screlu(result) : result;
    }
}

float forward_float_simd(const NKNNModel& model,
                         const std::vector<int>& features_white,
                         const std::vector<int>& features_black,
                         int stm) {
    std::array<float, L1_OUTPUT> acc_white, acc_black;
    
    // L1: SIMD sparse accumulation
    sparse_accum_simd(model, features_white, acc_white);
    sparse_accum_simd(model, features_black, acc_black);
    
    // Concatenate with SCReLU based on STM (SIMD)
    alignas(32) std::array<float, L2_INPUT> hidden1;
    
    if (stm == 0) {  // White to move
        for (int i = 0; i < L1_OUTPUT; i += 8) {
            __m256 w = _mm256_loadu_ps(&acc_white[i]);
            __m256 b = _mm256_loadu_ps(&acc_black[i]);
            _mm256_storeu_ps(&hidden1[i], screlu_avx(w));
            _mm256_storeu_ps(&hidden1[L1_OUTPUT + i], screlu_avx(b));
        }
    } else {  // Black to move
        for (int i = 0; i < L1_OUTPUT; i += 8) {
            __m256 w = _mm256_loadu_ps(&acc_white[i]);
            __m256 b = _mm256_loadu_ps(&acc_black[i]);
            _mm256_storeu_ps(&hidden1[i], screlu_avx(b));
            _mm256_storeu_ps(&hidden1[L1_OUTPUT + i], screlu_avx(w));
        }
    }
    
    // L2: Dense 512->32
    alignas(32) std::array<float, L2_OUTPUT> hidden2;
    dense_layer_simd(L2_INPUT, L2_OUTPUT, model.W2_f.data(), model.B2_f.data(),
                     hidden1.data(), hidden2.data(), true);
    
    // L3: Dense 32->32
    alignas(32) std::array<float, L3_OUTPUT> hidden3;
    dense_layer_simd(L2_OUTPUT, L3_OUTPUT, model.W3_f.data(), model.B3_f.data(),
                     hidden2.data(), hidden3.data(), true);
    
    // L4: Dense 32->1 (no activation, use SIMD dot product)
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < L3_OUTPUT; i += 8) {
        __m256 h = _mm256_loadu_ps(&hidden3[i]);
        __m256 w = _mm256_loadu_ps(&model.W4_f[i]);
        sum = _mm256_fmadd_ps(h, w, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    return _mm_cvtss_f32(sum128) + model.B4_f;
}
