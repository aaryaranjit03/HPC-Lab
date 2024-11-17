#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int send_array[2];  // Array of two integers to send
    int recv_array[2];  // Array to receive data at the root process

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get the total number of processes

    // Initialize the send_array with two numbers in each process
    send_array[0] = rank * 2;   // First element, e.g., 0, 2, 4, 6, ...
    send_array[1] = rank * 2 + 1; // Second element, e.g., 1, 3, 5, 7, ...

    if (rank != 0) {
        // Send the array of two integers to the root process (rank 0)
        MPI_Send(send_array, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Process %d sent: [%d, %d]\n", rank, send_array[0], send_array[1]);
    } else {
        // Root process (rank 0) receives data from all other processes
        for (int i = 1; i < size; i++) {
            MPI_Recv(recv_array, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Root process (rank 0) received from process %d: [%d, %d]\n", i, recv_array[0], recv_array[1]);
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
