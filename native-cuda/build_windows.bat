@echo off
REM NNUE CUDA Kernels Build Script
REM Requires: Visual Studio 2022 Preview, CUDA Toolkit 12.x

setlocal enabledelayedexpansion

REM Setup VS environment
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not initialize VS2022 environment
    exit /b 1
)

REM Check nvcc
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found in PATH
    exit /b 1
)

REM Create output directory
if not exist "build" mkdir build

REM Compile to DLL
echo Compiling CUDA kernels...
nvcc -shared -o build\nnue_cuda.dll nnue_kernels.cu ^
    -O3 ^
    -arch=sm_75 ^
    -gencode=arch=compute_75,code=sm_75 ^
    -gencode=arch=compute_86,code=sm_86 ^
    -gencode=arch=compute_89,code=sm_89 ^
    -Xcompiler "/MD /O2"

if errorlevel 1 (
    echo ERROR: Compilation failed
    exit /b 1
)

echo.
echo SUCCESS: build\nnue_cuda.dll created
dir build\nnue_cuda.dll

exit /b 0
