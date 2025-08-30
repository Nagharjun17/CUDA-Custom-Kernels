#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include "matmul.cuh"

const int MAX_NUM = 100;
const int MIN_NUM = -100;

int main(int argc, char const *argv[]){
    int M = 512;
    int N = 512;
    int K = 512;

    float* A = (float*)malloc(M*K*sizeof(float));
    float* B = (float*)malloc(K*N*sizeof(float));
    float* C = (float*)malloc(M*N*sizeof(float));

    for(int i=0;i<M*K;i++){
        A[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }
    for(int i=0;i<K*N;i++){
        B[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    float* d_A, d_B, d_C;

    cudaError_t error_A = cudaMalloc((void**)&d_A, M*K*sizeof(float));
    CUDA_CHECK(error_A);
    cudaError_t error_B = cudaMalloc((void**)&d_B, K*N*sizeof(float));
    CUDA_CHECK(error_B);
    cudaError_t error_C = cudaMalloc((void**)&d_C, M*N*sizeof(float));
    CUDA_CHECK(error_C);

    cudaError_t h2d_A = cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(h2d_A);
    cudaError_t h2d_B = cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(h2d_B);

    matrix_multiplication_tiled(d_A, d_B, d_C, M, N, K);

    cudaError_t d2h_C = cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(d2h_C);
}