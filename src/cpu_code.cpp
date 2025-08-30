#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>

using namespace std;

static const int N = 1024;

void vector_add(const vector<float>& A, const vector<float>& B, vector<float>& C){
    for(int i=0;i<N;i++){
        C[i] = A[i] + B[i];
    }
}

void vector_multiply(const vector<float>& A, const vector<float>& B, const int Q, const int R, const int S, vector<float>& C){
    for(int i=0;i<Q;i++){
        for(int j=0;j<R;j++){
            int idx = i*R + j;
            float acc = 0.0f;
            for(int k=0;k<S;k++){
                acc += A[i*S + k] * B[k*R + j];
            }
            C[idx] = acc;
        }
    }
}

int main(){
    vector<float> A(N);
    vector<float> B(N);
    vector<float> C(N);

    for(int i=0;i<N;i++){
        A[i] = sin(i);
        B[i] = cos(i);
    }

    vector_add(A, B, C);

    vector<float> X(N*N);
    vector<float> Y(N*N);
    vector<float> Z(N*N);

    for(int i=0;i<N*N;i++){
        X[i] = sin(i);
        Y[i] = cos(i);
    }

    vector_multiply(X, Y, N, N, N, Z);

    return 0;
}

// (0,0) (0,1) (0,2) (0,3)    (0,0) (0,1) (0,2) (0,3)
// (1,0) (1,1) (1,2) (1,3)    (1,0) (1,1) (1,2) (1,3)
// (2,0) (2,1) (2,2) (2,3)    (2,0) (2,1) (2,2) (2,3)
// (3,0) (3,1) (3,2) (3,3)    (3,0) (3,1) (3,2) (3,3)