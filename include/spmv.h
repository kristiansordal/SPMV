#pragma once
#include "graph.h"

// Taken from https://github.com/KennethLangedal/INF339/
typedef struct {
    int *send_count, *recv_count;
    int **send_items, **recv_items;
    double **send_lists, **recv_lists;
} comm_lists;

comm_lists init_comm_lists(int size);

void free_comm_lists(comm_lists *c, int size);

void find_sendlists(Graph M, int *p, int rank, int size, comm_lists c);

void find_recvlists(Graph M, int *p, int rank, int size, comm_lists c);

void exchange_separators(comm_lists c, double *y, int rank, int size);

void partition_graph(Graph G, int k, int *p, double *x);
