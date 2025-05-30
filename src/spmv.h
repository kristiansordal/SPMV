#pragma once
#include "mtx.h"
typedef struct {
    int send_count_total, receive_count_total;
    int *send_mark, *receive_mark;
    int *send_count, *receive_count;
    int **send_items, **receive_items;
    double **send_lists, **receive_lists;
} comm_lists;

void spmv(CSR g, double *x, double *y, long long int *flops);

void spmv_part(CSR g, int rank, int s, int t, double *x, double *y);

void partition_graph_1b(CSR g, int k, int *p, comm_lists *c);

void partition_graph_1c(CSR g, int k, int *p, comm_lists *c);

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c);

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c);

void partition_graph(CSR g, int num_partitions, int *partition_idx);

void partition_graph_naive(CSR g, int s, int t, int k, int *p);

void distribute_graph(CSR *g, int rank);

comm_lists init_comm_lists(int size);

void free_comm_lists(comm_lists *c, int size);

// void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c);

// void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c);

void reorder_separators(CSR g, int num_partitions, int *partition_idx, double *x, comm_lists *c);

void exchange_separators(comm_lists c, double *y, int *displs, int rank, int size);

// void exchange_separators(comm_lists c, double *x, double *y, int *displs, int rank, int size);

void exchange_required_separators(comm_lists c, double *y, int rank, int size);
