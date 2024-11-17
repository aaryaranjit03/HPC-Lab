#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, n = 10000;
    int local_sum = 0, total_sum = 0;
    int chunk_size, start, end;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank (ID) of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the chunk size for each process
    chunk_size = n / size;

    // Determine the range of integers for each process
    start = rank * chunk_size + 1;
    end = (rank == size - 1) ? n : (rank + 1) * chunk_size;  // Ensure last process covers all numbers

    // Compute the local sum for the process
    for (int i = start; i <= end; i++) {
        local_sum += i;
    }

    // Print local sum for debugging purposes
    printf("Process %d: Local sum = %d\n", rank, local_sum);

    // Ring Communication: Send local sum to next process, and receive from previous process
    int left = (rank == 0) ? size - 1 : rank - 1;   // Left neighbor in the ring
    int right = (rank == size - 1) ? 0 : rank + 1;  // Right neighbor in the ring

    // First step: Send the local sum to the next process
    MPI_Send(&local_sum, 1, MPI_INT, right, 0, MPI_COMM_WORLD);

    // Then, receive the partial sum from the previous process
    MPI_Recv(&total_sum, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Add the received sum to the local sum
    total_sum += local_sum;

    // The root process (rank 0) prints the final sum
    if (rank == 0) {
        printf("Total sum of first %d integers is: %d\n", n, total_sum);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
