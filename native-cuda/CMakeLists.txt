cmake_minimum_required(VERSION 3.21)
project(nnue_infer_cuda LANGUAGES CXX CUDA)

add_library(nnue_infer_cuda STATIC dummy.cu)
set_target_properties(nnue_infer_cuda PROPERTIES CUDA_ARCHITECTURES "89")
