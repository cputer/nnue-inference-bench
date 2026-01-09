# Build C++ baseline benchmark
$ErrorActionPreference = "Stop"

# Find MSVC
$vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build'
if (!(Test-Path $vsPath)) {
    $vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build'
}
if (!(Test-Path $vsPath)) {
    $vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build'
}
if (!(Test-Path $vsPath)) {
    $vsPath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build'
}

Write-Host "Using MSVC from: $vsPath"

# Create build directory
New-Item -ItemType Directory -Force -Path build | Out-Null

# Setup environment and compile
$vcvars = Join-Path $vsPath "vcvars64.bat"
$src = "nnue_cpu.cpp bench_main.cpp"

# Use cmd to run vcvars then compile
$cmd = @"
call "$vcvars" && cl /nologo /O2 /EHsc /std:c++17 /Fe:build\bench_cpp.exe $src
"@

cmd /c $cmd

if (Test-Path "build\bench_cpp.exe") {
    Write-Host "Build successful: build\bench_cpp.exe"
    Write-Host ""
    Write-Host "Run with: .\build\bench_cpp.exe --model ..\models\nikola_d12v2_gold.nknn"
} else {
    Write-Host "Build failed!"
    exit 1
}
