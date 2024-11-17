#include <stdio.h>
#include <omp.h>

int main() {
    // Set the number of threads to 2 (one for each series)
    omp_set_num_threads(2);

    // Parallel region where each thread will execute different tasks
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        // If thread_id is 0, print the series of 2
        if (thread_id == 0) {
            for (int i = 1; i <= 5; i++) { // Print 5 terms of the series 2, 2, 2, ...
                printf("Thread %d: %d\n", thread_id, 2);
            }
        }
        // If thread_id is 1, print the series of 4
        else if (thread_id == 1) {
            for (int i = 1; i <= 5; i++) { // Print 5 terms of the series 4, 4, 4, ...
                printf("Thread %d: %d\n", thread_id, 4);
            }
        }
    }

    return 0;
}
