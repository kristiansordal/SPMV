#pragma once
#include <stdio.h>

typedef struct {
    int n, m, nnz;
    int *vertices, *edges;
    double *vals;
} Graph;

Graph parse_and_validate_mtx(const char *path);

Graph parse_mtx(FILE *f);

void free_graph(Graph *G);

void sort_edges(Graph G);

void normalize_graph(Graph G);

int validate_graph(Graph G);
