#pragma once
struct CSRMatrix {
    int n;
    int num_rows;
    int num_cols;
    int local_nnz;
    int num_nonzeros;
    int *row_ptr; // TODO: Why is this not a pointer to pointers? Could it be faster?
    int *col_ptr;
    double *vals;
};
