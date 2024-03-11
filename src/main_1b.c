#include <graph.h>
#include <mpi.h>
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
    }

    fclose(file_in);
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *v_old, *v_new;
    comm_lists comm = init_comm_lists(size);
    Graph G;
    int *p = malloc(sizeof(int) * (size + 1)); // pfs array

    // read file
    if (rank == 0) {
        read_file(&G, argv);
        for (int i = 0; i < G.n; i++) {
            v_old[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
        }
        // for (int i = 0; i < G.n; i++) {
        //     printf("%d ", G.vertices[i]);
        // }
        // printf("\n");
        partition_graph(G, size, p, v_old);
        // for (int i = 0; i < G.n; i++) {
        //     printf("%d ", G.vertices[i]);
        // }
        // // printf("\n");
        // // for (int i = 0; i < size; i++) {
        // //     printf("%d ", p[i]);
        // // }
        // printf("\n");
    }

    MPI_Finalize();
    return 0;
}
