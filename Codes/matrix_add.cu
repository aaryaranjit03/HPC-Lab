#include <stdio.h>
#include <cuda.h>

// CUDA kernel for matrix addition
__global__ void matrixAddKernel(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = 3, cols = 3;
    int size = rows * cols * sizeof(int);

    // Host matrices
    int h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_B[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int h_C[rows * cols];

    // Device matrices
    int *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(3, 3); // 3x3 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result matrix C (A + B):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_C[i * cols + j]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
