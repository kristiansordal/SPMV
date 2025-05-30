#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSR g;
    int *p = malloc(sizeof(int) * (size + 1));

    for (int i = 0; i < size + 1; i++)
        p[i] = 0;

    comm_lists c = init_comm_lists(size);

    double tcomm, tcomp, t0, t1;

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);
        partition_graph_1b(g, size, p, &c);
    }

    distribute_graph(&g, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(c.send_count, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 2.0;
        y[i] = 2.0;
    }

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = p[i + 1] - p[i];
        displs[i] = p[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();
    for (int i = 0; i < 100; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        if (size == 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        } else {
            MPI_Allgatherv(y + displs[rank], c.send_count[rank], MPI_DOUBLE, y, c.send_count, displs, MPI_DOUBLE,
                           MPI_COMM_WORLD);
        }
        double tc2 = MPI_Wtime();
        double *tmp = y;
        y = x;
        x = tmp;
        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        double tc3 = MPI_Wtime();
        tcomm += tc2 - tc1;
        tcomp += tc3 - tc2;
    }

    t1 = MPI_Wtime();

    MPI_Allgatherv(y + displs[rank], recvcounts[rank], MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    double *tmp = x;
    x = y;
    y = tmp;

    long double comm_size = ((double)c.send_count[rank] * (size - 1) * 100.0 * 64.0) / (1024.0 * 1024.0 * 1024.0);

    long double max_comm_size = 0.0;
    long double min_comm_size = 0.0;
    long double avg_comm_size = 0.0;
    long double total_flops = 0.0;

    MPI_Reduce(&comm_size, &max_comm_size, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &min_comm_size, 1, MPI_LONG_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &avg_comm_size, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_comm_size /= size;

    double ops = (long long)g.num_cols * 2ll * 100ll;
    double time = t1 - t0;
    double l2 = 0.0;

    if (rank == 0) {
        for (int j = 0; j < g.num_rows; j++)
            l2 += x[j] * x[j];
        l2 = sqrt(l2);
    }

    if (rank == 0) {
        printf("Total time = %lfs\n", time);
        printf("Communication time = %lfs\n", tcomm);
        printf("Computation time = %lfs\n", tcomp);
        printf("NFLOPS = %lf\n", ops);
        printf("GFLOPS = %lf\n", ops / (time * 1e9));
        printf("Comm min = %Lf GB\nComm max = %Lf GB\nComm avg = %Lf GB\n", min_comm_size, max_comm_size,
               avg_comm_size);
        fflush(stdout);
    }

    free(y);
    free(x);
    free(p);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
