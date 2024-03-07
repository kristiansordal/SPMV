#include "read_file.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
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
