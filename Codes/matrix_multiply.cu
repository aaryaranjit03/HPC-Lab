#include <stdio.h>
#include <cuda.h>

__global__ void matrixMultiplyKernel(int *A, int *B, int *C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        int sum = 0;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

int main() {
    int A_rows = 2, A_cols = 3, B_cols = 2;

    int A[6] = {1, 2, 3, 4, 5, 6};
    int B[6] = {7, 8, 9, 10, 11, 12};
    int C[4];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_rows * A_cols * sizeof(int));
    cudaMalloc(&d_B, A_cols * B_cols * sizeof(int));
    cudaMalloc(&d_C, A_rows * B_cols * sizeof(int));

    cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, A_cols * B_cols * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B_cols + 15) / 16, (A_rows + 15) / 16);

    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);

    cudaMemcpy(C, d_C, A_rows * B_cols * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%d ", C[i * B_cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
