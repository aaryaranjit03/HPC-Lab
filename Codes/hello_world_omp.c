#include <stdio.h>
#include <omp.h>

int main() {
    // Start parallel region
    #pragma omp parallel
    {
        // Get the thread ID of the current thread
        int thread_id = omp_get_thread_num();
        
        // Print Hello World along with the thread ID
        printf("Hello, World! from thread %d\n", thread_id);
    }

    return 0;
}
