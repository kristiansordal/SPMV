#include "matrix.h"
#include "read_file.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STEPS 100

void read_file(struct CSRMatrix *M, char **argv) {
    const char *csr_file_name = argv[1];
    FILE *file_in = fopen(csr_file_name, "rb");

    if (!file_in) {
        perror("Error opening file");
    }

    if (fread(&M->n, sizeof(M->n), 1, file_in) != 1 || M->n <= 0 ||
        fread(&M->num_cols, sizeof(M->num_cols), 1, file_in) != 1 || M->num_cols <= 0 ||
        fread(&M->num_nonzeros, sizeof(M->num_nonzeros), 1, file_in) != 1 || M->num_nonzeros <= 0) {

        fprintf(stderr, "Error reading CSR header section from binary file %s\n", csr_file_name);
        fclose(file_in);
    }

    M->num_rows = M->n;
    M->row_ptr = (int *)malloc(sizeof(int) * (M->n + 1));
    M->col_ptr = (int *)malloc(sizeof(int) * M->num_nonzeros);
    M->vals = (double *)malloc(sizeof(double) * M->num_nonzeros);

    if (fread(M->row_ptr, sizeof(int), M->n + 1, file_in) != M->n + 1 ||
        fread(M->col_ptr, sizeof(int), M->num_nonzeros, file_in) != M->num_nonzeros ||
        fread(M->vals, sizeof(double), M->num_nonzeros, file_in) != M->num_nonzeros) {

        fprintf(stderr, "Error reading matrix data from binary file %s\n", csr_file_name);
        free(M->vals);
        free(M->row_ptr);
        free(M->col_ptr);
        fclose(file_in);
    }

    fclose(file_in);
}
int main(int argc, char **argv) {

    int rank, size;
    double t_file_read = 0, t_comp = 0, t_comm = 0, t_total, t = 0, t_start_comp = 0, t_end_comp = 0;
    struct CSRMatrix M;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    t_total = MPI_Wtime();
    double *v_old, *v_new;
    v_new = (double *)malloc(sizeof(double) * M.n);
    v_old = (double *)malloc(sizeof(double) * M.n);

    // Variables for scattering of row and column pointer.
    // NOTE Column send counts and column displacement is also used for the values
    int *row_send_counts = (int *)malloc(sizeof(int) * size);
    int *col_send_counts = (int *)malloc(sizeof(int) * size);
    int *row_displs = (int *)malloc(sizeof(int) * size);
    int *col_displs = (int *)malloc(sizeof(int) * size);
    memset(row_send_counts, 0, sizeof(int) * size);
    memset(col_send_counts, 0, sizeof(int) * size);
    memset(row_displs, 0, sizeof(int) * size);
    memset(col_displs, 0, sizeof(int) * size);

    if (rank == 0) {
        t_file_read = MPI_Wtime();
        read_file(&M, argv);
        t_file_read = MPI_Wtime() - t_file_read;

        v_old = (double *)malloc(sizeof(double) * M.n);
        for (int i = 0; i < M.n; i++)
            v_old[i] = 1;

        int rows_per_rank = M.n / size;
        int avg_nnz_per_rank = M.num_nonzeros / size;
        int curr_rank = 0;

        // Load balance
        for (int i = 0; i < M.n; i++) {
            row_send_counts[curr_rank]++;

            if (M.row_ptr[i + 1] > avg_nnz_per_rank * (curr_rank + 1)) {
                row_send_counts[curr_rank]++;
                curr_rank++;
            }
        }

        // Compute displacements for row pointers
        for (int i = 0; i < size; i++)
            row_displs[i] = i == 0 ? 0 : row_displs[i - 1] + row_send_counts[i - 1] - 1;

        // Compute send counts and displacements for column (and value) pointers
        for (int i = 0; i < size; i++) {
            col_send_counts[i] = i == size - 1 ? M.num_nonzeros - M.row_ptr[row_displs[i]]
                                               : M.row_ptr[row_displs[i + 1]] - M.row_ptr[row_displs[i]];
            col_displs[i] = i == 0 ? 0 : col_displs[i - 1] + col_send_counts[i - 1];
        }
    }

    MPI_Bcast(row_send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    M.num_rows = row_send_counts[rank];
    M.num_cols = col_send_counts[rank]; // local nnz

    if (rank != 0) {
        M.row_ptr = (int *)malloc(sizeof(int) * M.num_rows);
        M.col_ptr = (int *)malloc(sizeof(int) * M.num_cols);
        M.vals = (double *)malloc(sizeof(double) * M.num_cols);
    }

    // Scatter row column and value pointers
    MPI_Scatterv(M.row_ptr, row_send_counts, row_displs, MPI_INT, M.row_ptr, M.num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(M.col_ptr, col_send_counts, col_displs, MPI_INT, M.col_ptr, M.num_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(M.vals, col_send_counts, col_displs, MPI_DOUBLE, M.vals, M.num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Adjust row pointers to local indices
    for (int i = 1; i < M.num_rows; i++)
        M.row_ptr[i] -= M.row_ptr[0];
    M.row_ptr[0] = 0;

    --M.num_rows;

    // Broadcast v_old
    MPI_Bcast(&M.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        v_old = (double *)malloc(sizeof(double) * M.n);
        v_new = (double *)malloc(sizeof(double) * M.n);
    }

    MPI_Bcast(v_old, M.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(col_send_counts);
    free(col_displs);

    if (rank == 0) {
        M.row_ptr = (int *)realloc(M.row_ptr, sizeof(int) * M.num_rows);
        M.col_ptr = (int *)realloc(M.col_ptr, sizeof(int) * M.num_cols);
        M.vals = (double *)realloc(M.vals, sizeof(double) * M.num_cols);
    }

    for (int i = 0; i < size; i++)
        row_send_counts[i]--;

    for (int i = 0; i < 100; i++) {
        if (rank == 0)
            t_start_comp = MPI_Wtime();

        for (int row = 0; row < M.num_rows - 1; row++)
            for (int col = M.row_ptr[row]; col < M.row_ptr[row + 1]; col++)
                v_new[row] += M.vals[col] * v_old[M.col_ptr[col]];

        if (rank == 0) {
            t_end_comp = MPI_Wtime();
            t_comp += t_end_comp - t_start_comp;
            t = MPI_Wtime();
        }

        MPI_Allgatherv(v_new, M.num_rows - 1, MPI_DOUBLE, v_old, row_send_counts, row_displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (rank == 0) {
            t = MPI_Wtime() - t;
            t_comm += t;
        }
    }

    if (rank == 0) {
        t_comp = MPI_Wtime() - t_comp;
        t_total = MPI_Wtime() - t_total;

        unsigned long long FLOPS = (unsigned long long)M.num_nonzeros * STEPS * 2;

        printf("TIME READ  : %f\n", t_file_read);
        printf("TIME COMP  : %f\n", t_comp);
        printf("TIME COMM  : %f\n", t_comm);
        printf("TIME TOTAL : %f\n", t_total);
        printf("FLOPS      : %llu\n", FLOPS);
        printf("GLOPS      : %f\n", (FLOPS / 1e9) / t_total);
    }

    free(v_old);
    free(v_new);
    free(row_send_counts);
    free(row_displs);
    free(M.row_ptr);
    free(M.col_ptr);
    free(M.vals);

    MPI_Finalize();

    return 0;
}
