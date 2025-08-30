#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "matmul.cuh"

const int MAX_NUM = 100;
const int MIN_NUM = -100;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct CudaTimer {
    cudaEvent_t begin{}, end{};
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(begin));
    }
    float stop() {
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time, begin, end));
        return time;
    }
};

template <typename Launch>
float bench_kernel(Launch launch, const char* name, const float* dA, const float* dB, float* dC, int M, int N, int K, int warmup=3, int repeat=10, const char* csv_path="scripts/results.csv") {
    for(int i=0;i<warmup;i++){
        launch(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float best_time = 1e30f;
    for(int r=0;r<repeat;r++){
        CudaTimer t;
        t.start();
        launch(dA, dB, dC, M, N, K);
        float time = t.stop();
        if(time < best_time){
            best_time = time;
        }
    }

    double performance = 2.0 * M * N * K / (best_time / 1e3) / 1e9; //GFLOPS
    std::printf("%-10s : %4dx%-4dx%-4d : %8.3f ms, %8.2f GFLOPS\n", name, M, N, K, best_time, performance);

    std::ofstream csv(csv_path, std::ios::app);
    if(csv.tellp() == 0){
        csv << "name,M,N,K,time,GFLOPS\n";
    }
    csv << name << "," << M << "," << N << "," << K << "," << best_time << "," << performance << "\n";
    csv.close();
    return best_time;
}

int main(int argc, char const *argv[]){
    int M = 1024;
    int N = 1024;
    int K = 1024;

    std::vector<float> A(M*K), B(K*N), C(M*N), C_ref(M*N);

    for(int i=0;i<M*K;i++){
        A[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }
    for(int i=0;i<K*N;i++){
        B[i] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
    }

    float *d_A, *d_B, *d_C;

    cudaError_t error_A = cudaMalloc((void**)&d_A, M*K*sizeof(float));
    CUDA_CHECK(error_A);
    cudaError_t error_B = cudaMalloc((void**)&d_B, K*N*sizeof(float));
    CUDA_CHECK(error_B);
    cudaError_t error_C = cudaMalloc((void**)&d_C, M*N*sizeof(float));
    CUDA_CHECK(error_C);

    cudaError_t h2d_A = cudaMemcpy(d_A, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(h2d_A);
    cudaError_t h2d_B = cudaMemcpy(d_B, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(h2d_B);

    bench_kernel(matrix_multiplication_naive, "Naive", d_A, d_B, d_C, M, N, K);
    bench_kernel(matrix_multiplication_tiled, "Tiled", d_A, d_B, d_C, M, N, K);

    cudaError_t d2h_C = cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(d2h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}