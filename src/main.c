#include "matrix.h"
#include "read_file.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define STEPS 100

int main(int argc, char *argv[]) {

    int rank, size;
    struct CSRMatrix M;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *v_old, *v_new;

    // Variables for scattering of row and column pointer.
    // NOTE Column send counts and column displacement is also used for the values
    int *row_send_counts = (int *)malloc(sizeof(int) * size);
    int *col_send_counts = (int *)malloc(sizeof(int) * size);
    int *row_displs = (int *)malloc(sizeof(int) * size);
    int *col_displs = (int *)malloc(sizeof(int) * size);

    row_displs[0] = 0;
    col_displs[0] = 0;

    if (rank == 0) {
        read_file(&M, argv);
        for (int i = 0; i < M.num_nonzeros; i++) {
            printf("%d, %f\n", i, M.vals[i]);
        }
        printf("\n");

        // allocate memory for v_old and v_new
        v_old = (double *)malloc(sizeof(double) * M.n);
        v_new = (double *)malloc(sizeof(double) * M.n);

        for (int i = 0; i < M.n; i++) {
            v_old[i] = rand() % 5;
        }

        int rows_per_rank = M.n / size;
        int avg_nnz_per_rank = M.num_nonzeros / size;
        int curr_rank = 0;

        // Load balance
        for (int i = 0; i < M.num_rows; i++) {
            row_send_counts[curr_rank]++;

            if (M.row_ptr[i + 1] > avg_nnz_per_rank * (curr_rank + 1)) {
                row_send_counts[curr_rank]++;
                curr_rank++;
            }
        }

        // Compute displacements for row pointers
        for (int i = 0; i < size; i++) {
            // -1 because of the overlap
            row_displs[i] = i == 0 ? 0 : row_displs[i - 1] + row_send_counts[i - 1] - 1;
        }

        // Compute send counts and displacements for column (and value) pointers
        for (int i = 0; i < size; i++) {

            // We need to send #values at row i - #values at row i-1, except for last value, here we use num nonzeros
            // instead
            col_send_counts[i] = i == size - 1 ? M.num_nonzeros - M.row_ptr[row_displs[i]]
                                               : M.row_ptr[row_displs[i + 1]] - M.row_ptr[row_displs[i]];

            col_displs[i] = i == 0 ? 0 : col_displs[i - 1] + col_send_counts[i - 1];
        }
    }

    // Broadcast send_counts and displs to all ranks
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

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            printf("%d, %d, %d\n", rank, col_displs[i], col_send_counts[i]);
        }
    }

    // Scatter row column and value pointers
    MPI_Scatterv(M.row_ptr, row_send_counts, row_displs, MPI_INT, M.row_ptr, M.num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(M.col_ptr, col_send_counts, col_displs, MPI_INT, M.col_ptr, M.num_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(M.vals, col_send_counts, col_displs, MPI_DOUBLE, M.vals, M.num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast v_old
    MPI_Bcast(&M.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        v_old = (double *)malloc(sizeof(double) * M.n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(v_old, M.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for v_new
    v_new = (double *)malloc(sizeof(double) * M.num_rows);

    for (int i = 0; i < M.num_rows; i++) {
        v_new[i] = 0;
    }

    // Free send counts and displacements arrays
    free(col_send_counts);
    free(col_displs);

    // Resize rank 0s matrix - it doesn't need to store the now scattered values
    if (rank == 0) {
        M.row_ptr = (int *)realloc(M.row_ptr, sizeof(int) * M.num_rows);
        M.col_ptr = (int *)realloc(M.col_ptr, sizeof(int) * M.num_cols);
        M.vals = (double *)realloc(M.vals, sizeof(double) * M.num_cols);
    }

    for (int i = 0; i < STEPS; i++) {
        for (int row = 0; row < M.num_rows - 1; row++) {
            for (int col = M.row_ptr[row] - M.row_ptr[0]; col < M.row_ptr[row + 1] - M.row_ptr[0]; col++) {
                v_new[row] += M.vals[col] * v_old[M.col_ptr[col]];
            }
        }

        MPI_Allgatherv(v_new, M.num_rows, MPI_DOUBLE, v_old, row_send_counts, row_displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
