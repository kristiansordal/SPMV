#include <mpi.h>
#include <mtx.h>
#include <spmv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void read_file(Graph *G, char **argv) {
    const char *csr_file_name = argv[1];
    FILE *file_in = fopen(csr_file_name, "rb");

    if (!file_in)
        perror("Error opening file");

    if (fread(&G->n, sizeof(G->n), 1, file_in) != 1 || G->n <= 0 || fread(&G->m, sizeof(G->m), 1, file_in) != 1 ||
        G->m <= 0 || fread(&G->nnz, sizeof(G->nnz), 1, file_in) != 1 || G->nnz <= 0) {
        fprintf(stderr, "Error reading CSR header section from binary file %s\n", csr_file_name);
        fclose(file_in);
    } else {
        printf("Successfully read CSR header section from binary file %s\n", csr_file_name);
    }

    G->vertices = (int *)malloc(sizeof(int) * (G->n + 1));
    G->m = G->nnz;
    printf("Num nonzeros: %d\n", G->nnz);
    G->edges = (int *)malloc(sizeof(int) * G->nnz);
    G->vals = (double *)malloc(sizeof(double) * G->nnz);

    if (fread(G->vertices, sizeof(int), G->n + 1, file_in) != G->n + 1 ||
        fread(G->edges, sizeof(int), G->nnz, file_in) != G->nnz ||
        fread(G->vals, sizeof(double), G->nnz, file_in) != G->nnz) {

        fprintf(stderr, "Error reading matrix data from binary file %s\n", csr_file_name);
        free(G->vals);
        free(G->vertices);
        free(G->edges);
        fclose(file_in);
    } else {
        printf("Successfully read matrix data from binary file %s\n", csr_file_name);
        printf("|V| = %d |E| = %d NNZ: %d\n", G->n, G->m, G->nnz);
    }

    fclose(file_in);
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *v_old, *v_new;
    comm_lists_singleton comm = init_comm_list_singleton(size);
    Graph G;
    int *p = malloc(sizeof(int) * (size + 1)); // pfs array

    // read file on rank 0
    if (rank == 0) {
        G = parse_and_validate_mtx(argv[1]);
        // read_file(&G, argv);
        v_old = malloc(sizeof(double) * G.n);
        for (int i = 0; i < G.n; i++)
            v_old[i] = 1;

        partition_graph(G, size, p, v_old);
    }

    distribute_graph(&G, rank);

    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);
    find_sendlists_singleton(G, p, rank, size, &comm);

    // if (rank == 0) {
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            for (int i = 0; i < comm.send_count; i++) {
                printf("rank: %d, comm.send_lists[%d]: %d\n", rank, i, comm.send_items[i]);
            }
        }
    }
    // }

    v_new = malloc(sizeof(double) * G.n);

    free_graph(&G);
    free(p);
    if (rank == 0)
        free(v_old);
    free(v_new);
    free_comm_lists_singleton(&comm);
    MPI_Finalize();
    return 0;
}
