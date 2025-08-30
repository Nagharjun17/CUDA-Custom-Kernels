#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vec_add(const float *A, const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

void vector_addition(const float *A, const float *B, float *C, int N){
    dim3 num_threads(256);
    dim3 num_blocks((N + num_threads.x - 1)/num_threads.x);

    vec_add<<<num_blocks, num_threads>>>(A, B, C, N);
}