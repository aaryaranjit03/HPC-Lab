#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, N = 10000;
    int local_sum = 0, total_sum = 0;
    int chunk_size, start, end;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank (ID) and the size (total number of processes)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the chunk size for each process
    chunk_size = N / size;

    // Each process calculates the start and end index for its range of numbers
    start = rank * chunk_size + 1;
    end = (rank == size - 1) ? N : (rank + 1) * chunk_size;  // Ensure last process covers all numbers

    // Compute the local sum for this process's range
    for (int i = start; i <= end; i++) {
        local_sum += i;
    }

    // Print local sum from each process (for debugging purposes)
    printf("Process %d: Local sum = %d\n", rank, local_sum);

    // Use MPI_Reduce to sum all the local sums into the total_sum at root process
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // The root process prints the final sum
    if (rank == 0) {
        printf("Total sum of first %d integers: %d\n", N, total_sum);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
