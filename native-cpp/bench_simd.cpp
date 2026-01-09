// bench_simd.cpp - Benchmark for SIMD-optimized NNUE (Mind-equivalent)

#include "nnue_cpu.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>

// Forward declaration of SIMD implementation
float forward_float_simd(const NKNNModel& model,
                         const std::vector<int>& features_white,
                         const std::vector<int>& features_black,
                         int stm);

// MD5 implementation (same as bench_main.cpp)
namespace md5 {
    typedef uint32_t u32;
    typedef uint8_t u8;
    
    static const u32 S[] = {7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                            5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                            4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                            6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
    static const u32 K[] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391};
    
    inline u32 rotl(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
    
    void hash(const u8* msg, size_t len, u8 out[16]) {
        u32 a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
        size_t newlen = ((len + 8) / 64 + 1) * 64;
        std::vector<u8> buf(newlen, 0);
        memcpy(buf.data(), msg, len);
        buf[len] = 0x80;
        uint64_t bits = len * 8;
        memcpy(&buf[newlen - 8], &bits, 8);
        
        for (size_t chunk = 0; chunk < newlen; chunk += 64) {
            u32 M[16];
            memcpy(M, &buf[chunk], 64);
            u32 A = a0, B = b0, C = c0, D = d0;
            for (int i = 0; i < 64; i++) {
                u32 F, g;
                if (i < 16) { F = (B & C) | (~B & D); g = i; }
                else if (i < 32) { F = (D & B) | (~D & C); g = (5*i + 1) % 16; }
                else if (i < 48) { F = B ^ C ^ D; g = (3*i + 5) % 16; }
                else { F = C ^ (B | ~D); g = (7*i) % 16; }
                F = F + A + K[i] + M[g];
                A = D; D = C; C = B; B = B + rotl(F, S[i]);
            }
            a0 += A; b0 += B; c0 += C; d0 += D;
        }
        memcpy(out, &a0, 4); memcpy(out+4, &b0, 4);
        memcpy(out+8, &c0, 4); memcpy(out+12, &d0, 4);
    }
}

uint32_t compute_checksum_md5(const std::vector<float>& evals) {
    std::vector<uint8_t> data(evals.size() * 4);
    memcpy(data.data(), evals.data(), data.size());
    uint8_t digest[16];
    md5::hash(data.data(), data.size(), digest);
    uint32_t result = 0;
    for (int i = 0; i < 4; i++) {
        result = (result << 8) | digest[i];
    }
    return result;
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/nikola_d12v2_gold.nknn";
    int batch_size = 1000;
    int warmup_iters = 10;
    int measured_iters = 50;
    int seed = 42;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i+1 < argc) model_path = argv[++i];
        else if ((arg == "--batch" || arg == "-b") && i+1 < argc) batch_size = std::stoi(argv[++i]);
        else if ((arg == "--warmup" || arg == "-w") && i+1 < argc) warmup_iters = std::stoi(argv[++i]);
        else if ((arg == "--iters" || arg == "-i") && i+1 < argc) measured_iters = std::stoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc) seed = std::stoi(argv[++i]);
    }
    
    std::cerr << "Loading model: " << model_path << std::endl;
    NKNNModel model;
    if (!load_nknn(model_path, model)) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }
    std::cerr << "Model loaded (SIMD mode)" << std::endl;
    
    std::cerr << "Generating " << batch_size << " positions (seed=" << seed << ")..." << std::endl;
    std::vector<Position> positions(batch_size);
    for (int i = 0; i < batch_size; i++) {
        positions[i] = create_random_position(seed + i);
    }
    
    std::vector<std::vector<int>> features_w(batch_size), features_b(batch_size);
    for (int i = 0; i < batch_size; i++) {
        extract_halfkp_features(positions[i], features_w[i], features_b[i]);
    }
    
    std::cerr << "Warmup: " << warmup_iters << " iterations..." << std::endl;
    for (int iter = 0; iter < warmup_iters; iter++) {
        for (int i = 0; i < batch_size; i++) {
            volatile float e = forward_float_simd(model, features_w[i], features_b[i], positions[i].stm);
            (void)e;
        }
    }
    
    std::cerr << "Measured: " << measured_iters << " iterations..." << std::endl;
    std::vector<double> times_ms;
    std::vector<float> all_evals;
    
    for (int iter = 0; iter < measured_iters; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> evals(batch_size);
        for (int i = 0; i < batch_size; i++) {
            evals[i] = forward_float_simd(model, features_w[i], features_b[i], positions[i].stm);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_ms.push_back(ms);
        
        if (iter == 0) all_evals = evals;
    }
    
    std::sort(times_ms.begin(), times_ms.end());
    double p50 = times_ms[times_ms.size() / 2];
    double p95 = times_ms[static_cast<size_t>(times_ms.size() * 0.95)];
    double mean = 0;
    for (double t : times_ms) mean += t;
    mean /= times_ms.size();
    double throughput = (batch_size / (p50 / 1000.0));
    
    uint32_t checksum = compute_checksum_md5(all_evals);
    
    std::cout << "{" << std::endl;
    std::cout << "  \"implementation\": \"Mind CPU (SIMD)\"," << std::endl;
    std::cout << "  \"device\": \"CPU\"," << std::endl;
    std::cout << "  \"tier\": \"B\"," << std::endl;
    std::cout << "  \"batch_size\": " << batch_size << "," << std::endl;
    std::cout << "  \"warmup_iters\": " << warmup_iters << "," << std::endl;
    std::cout << "  \"measured_iters\": " << measured_iters << "," << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  \"p50_ms\": " << p50 << "," << std::endl;
    std::cout << "  \"p95_ms\": " << p95 << "," << std::endl;
    std::cout << "  \"mean_ms\": " << mean << "," << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  \"throughput_pos_per_s\": " << throughput << "," << std::endl;
    std::cout << "  \"checksum\": \"0x" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << checksum << "\"" << std::endl;
    std::cout << "}" << std::endl;
    
    std::cerr << "=== RESULTS ===" << std::endl;
    std::cerr << "p50: " << std::fixed << std::setprecision(3) << p50 << " ms" << std::endl;
    std::cerr << "Throughput: " << std::fixed << std::setprecision(0) << throughput << " pos/s" << std::endl;
    std::cerr << "Checksum: 0x" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << checksum << std::endl;
    
    return 0;
}
