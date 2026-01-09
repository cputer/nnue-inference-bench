# Build Mind CPU benchmark
# Requires: Mind compiler (mindc) in PATH

$ErrorActionPreference = "Stop"

Write-Host "Building Mind CPU NNUE benchmark..."

# Check for Mind compiler
$mindc = Get-Command mindc -ErrorAction SilentlyContinue
if (-not $mindc) {
    Write-Host "ERROR: Mind compiler (mindc) not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "To build Mind CPU benchmark, install the Mind compiler:"
    Write-Host "  1. Download from https://mind-lang.dev/download"
    Write-Host "  2. Add to PATH"
    Write-Host "  3. Re-run this script"
    Write-Host ""
    Write-Host "For now, use the C++ baseline (native-cpp/build.ps1)"
    exit 1
}

# Create build directory
New-Item -ItemType Directory -Force -Path build | Out-Null

# Compile Mind sources
Write-Host "Compiling nnue_model.mind..."
& mindc -O3 -target native -c nnue_model.mind -o build/nnue_model.o

Write-Host "Compiling nnue_infer.mind..."
& mindc -O3 -target native -simd avx2 -c nnue_infer.mind -o build/nnue_infer.o

Write-Host "Compiling bench_main.mind..."
& mindc -O3 -target native -simd avx2 bench_main.mind -o build/mind_nnue_bench.exe `
    build/nnue_model.o build/nnue_infer.o

if (Test-Path "build/mind_nnue_bench.exe") {
    Write-Host "Build successful: build/mind_nnue_bench.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run with: .\build\mind_nnue_bench.exe --model ..\models\nikola_d12v2_gold.nknn"
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
