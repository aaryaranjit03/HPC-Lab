#include <stdio.h>
#include <omp.h>

int main() {
    // Set the number of threads to 4
    omp_set_num_threads(4);

    // Parallel region where each thread will print your name
    #pragma omp parallel
    {
        // Each thread prints your name
        printf("Hello, my name is Sourav! Printed by thread %d\n", omp_get_thread_num());
    }

    return 0;
}
