#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int rank, size;
    FILE *fp;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process opens the file in append mode
    fp = fopen("/tmp/pids.txt", "a");
    if (fp == NULL) {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort if file cannot be opened
    }

    // Write the PID to the file
    fprintf(fp, "%d\n", getpid());
    fclose(fp);

    sleep(10);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Hello, world! I am master\n");
    }

    MPI_Finalize();
    return 0;
}
