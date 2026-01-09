# Build SIMD-optimized C++ benchmark (Mind-equivalent)
$ErrorActionPreference = "Stop"

$vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build'
if (!(Test-Path $vsPath)) {
    $vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build'
}

Write-Host "Using MSVC from: $vsPath"
New-Item -ItemType Directory -Force -Path build | Out-Null

$vcvars = Join-Path $vsPath "vcvars64.bat"

# Compile with AVX2 support
$cmd = @"
call "$vcvars" && cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /Fe:build\bench_simd.exe nnue_cpu.cpp nnue_cpu_simd.cpp bench_simd.cpp
"@

cmd /c $cmd

if (Test-Path "build\bench_simd.exe") {
    Write-Host "Build successful: build\bench_simd.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
