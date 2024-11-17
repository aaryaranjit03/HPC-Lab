#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get the total number of processes

    // Print "Hello World" along with the rank of the process
    printf("Hello World from process %d of %d\n", rank, size);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
