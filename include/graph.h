#pragma once
typedef struct {
    int n, m, nnz;
    int *vertices, *edges;
    double *vals;
} Graph;
