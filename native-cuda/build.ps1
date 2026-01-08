# NNUE CUDA Build Script
$ErrorActionPreference = 'Stop'

# Find MSVC
$vsPath = 'C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC'
$msvcDir = Get-ChildItem $vsPath | Select-Object -Last 1
$ccbin = Join-Path $msvcDir.FullName 'bin\Hostx64\x64'
Write-Host "Using MSVC: $ccbin"

# Set up paths
$cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:Path = "$cudaPath\bin;" + $env:Path

Set-Location 'C:\Users\Admin\projects\nnue-inference-bench\native-cuda'
if (-not (Test-Path 'build')) { New-Item -ItemType Directory -Path 'build' | Out-Null }

Write-Host 'Compiling CUDA kernels...'
$nvccArgs = @(
    '-shared',
    '-o', 'build\nnue_cuda.dll',
    'nnue_kernels.cu',
    '-O3',
    '-arch=sm_75',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_86,code=sm_86',
    '-Xcompiler', '/MD /O2',
    '-ccbin', $ccbin
)

& nvcc @nvccArgs 2>&1 | Write-Host

if ($LASTEXITCODE -eq 0) {
    Write-Host 'SUCCESS: DLL created'
    Get-Item 'build\nnue_cuda.dll'
} else {
    Write-Host 'FAILED: Compilation error'
    exit 1
}
