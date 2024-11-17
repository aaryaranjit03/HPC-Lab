#include <stdio.h>
#include <omp.h>

int main() {
    // Initialize OpenMP with 4 threads
    omp_set_num_threads(4);

    // Declare an integer variable val and initialize it to 1234
    int val = 1234;

    // Print the initial value of val outside the parallel region
    printf("Initial value of val (outside parallel region): %d\n", val);

    // Enter an OpenMP parallel region with the firstprivate clause applied to val
    #pragma omp parallel firstprivate(val)
    {
        // Get the thread number
        int thread_id = omp_get_thread_num();

        // Print the current value of val (private copy for each thread)
        printf("Thread %d: Current value of val = %d\n", thread_id, val);

        // Increment val by 1 (this only affects the private copy for this thread)
        val++;

        // Print the updated value of val (private copy after increment)
        printf("Thread %d: Updated value of val = %d\n", thread_id, val);
    }

    // Print the final value of val outside the parallel region
    printf("Final value of val (outside parallel region): %d\n", val);

    return 0;
}
