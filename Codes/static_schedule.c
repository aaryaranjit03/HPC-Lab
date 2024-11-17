#include <stdio.h>
#include <omp.h>

int main() {
    // Define the number of iterations in the loop
    int num_iterations = 100;

    // Define the chunk size
    int chunk = 10;

    // Initialize OpenMP with a specific number of threads (e.g., 4 threads)
    omp_set_num_threads(4);

    // Print a message indicating the start of parallel execution
    printf("Starting parallel loop with static scheduling and chunk size %d:\n", chunk);

    // Parallel for loop with static scheduling and chunk size
    #pragma omp parallel for schedule(static, chunk)
    for (int i = 0; i < num_iterations; i++) {
        // Get the thread ID
        int thread_id = omp_get_thread_num();

        // Print the iteration index and the corresponding thread ID
        printf("Thread %d: Iteration %d\n", thread_id, i);
    }

    // Print a message indicating the end of the parallel execution
    printf("Parallel loop execution completed.\n");

    return 0;
}
