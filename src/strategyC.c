#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSR g;
    int *p = malloc(sizeof(int) * (size + 1));
    for (int i = 0; i < size + 1; i++) {
        p[i] = 0;
    }

    comm_lists c = init_comm_lists(size);
    double tcomm, tcomp, t0, t1;

    for (int i = 0; i < size; i++) {
        c.send_items[i] = malloc(sizeof(int) * size);
        c.receive_items[i] = malloc(sizeof(int) * size);
    }

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);
        partition_graph_1c(g, size, p, &c);
    }

    for (int i = 0; i < size; i++) {
        MPI_Bcast(c.send_items[i], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(c.receive_items[i], size, MPI_INT, 0, MPI_COMM_WORLD);
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

    MPI_Barrier(MPI_COMM_WORLD);

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
        exchange_separators(c, y, displs, rank, size);
        double tc2 = MPI_Wtime();
        double *tmp = y;
        y = x;
        x = tmp;
        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        double tc3 = MPI_Wtime();
        tcomm += tc2 - tc1;
        tcomp += tc3 - tc2;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    MPI_Allgatherv(y + displs[rank], recvcounts[rank], MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    double *tmp = x;
    x = y;
    y = tmp;

    long double comm_size = 0.0;

    for (int i = 0; i < size; i++) {
        if (i != rank && c.send_items[rank][i] > 0) {
            comm_size += c.send_count[i];
        }
    }

    comm_size = (comm_size * 64.0 * 100.0) / (1024.0 * 1024.0 * 1024.0);

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

    // Print results
    if (rank == 0) {
        printf("Total time = %lfs\n", time);
        printf("Communication time = %lfs\n", tcomm);
        printf("Computation time = %lfs\n", tcomp);
        printf("GFLOPS = %lf\n", ops / (time * 1e9));
        printf("compGFLOPS = %Lf\n", total_flops / (time * 1e9));
        printf("NFLOPS = %lf\n", ops);
        printf("Comm min = %Lf GB\nComm max = %Lf GB\nComm avg = %Lf GB\n", min_comm_size, max_comm_size,
               avg_comm_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(y);
    free(x);
    free(p);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
