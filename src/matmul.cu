#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float agg = 0.0f;
        for(int x=0;x<K;x++){
            agg += A[row*K + x] * B[x*N + col];
        }
        C[row*N + col] = agg;
    }
}

void matrix_multiplication_naive(const float *A, const float *B, float *C, int M, int N, int K){
    dim3 num_threads(16, 16);
    dim3 num_blocks((N + num_threads.x - 1) / num_threads.x, (M + num_threads.y - 1) / num_threads.y);
    matmul_naive<<<num_blocks, num_threads>>>(A, B, C, M, N, K);
}

__global__ void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sh_A[16][16];
    __shared__ float sh_B[16][16];

    float agg = 0.0f;

    for(int tile=0;tile<(K+16-1)/16;tile++){
        if(row < M && tile*16 + threadIdx.x < K)
            sh_A[threadIdx.y][threadIdx.x] = A[row*K + tile*16 + threadIdx.x];
        else
            sh_A[threadIdx.y][threadIdx.x] = 0.0f;
        if(col < N && tile*16 + threadIdx.y < K)
            sh_B[threadIdx.y][threadIdx.x] = B[(tile*16 + threadIdx.y)*N + col];
        else
            sh_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for(int x=0;x<16;x++){
            agg += sh_A[threadIdx.y][x] * sh_B[x][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        C[row*N + col] = agg;
    }
}

void matrix_multiplication_tiled(const float *A, const float *B, float *C, int M, int N, int K){
    dim3 num_threads(16, 16);
    dim3 num_blocks((N + num_threads.x - 1) / num_threads.x, (M + num_threads.y - 1) / num_threads.y);
    matmul_tiled<<<num_blocks, num_threads>>>(A, B, C, M, N, K);
}