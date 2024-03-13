#pragma once
#include "mtx.h"

/* Taken from https://github.com/KennethLangedal/INF339/
 * Data structure for storing communication lists.
 * Used in the 1d communication scheme. Where each rank sends only
 * what is needed to each other rank.
 */
typedef struct {
    int *send_count, *recv_count;
    int **send_items, **recv_items;
    double **send_lists, **recv_lists;
} comm_lists;

/* Singleton version of communication scheme.
 * Used in communication strategy 1b: seperators are broadcasted
 * to each other rank, even though they may not need it.
 */
typedef struct {
    int send_count, recv_count;
    int *send_items, *recv_items;
    double *send_list, *recv_list;
} comm_lists_singleton;

comm_lists init_comm_lists(int size);
void free_comm_lists(comm_lists *c, int size);
void find_sendlists(Graph G, int *p, int rank, int size, comm_lists c);
void find_recvlists(Graph G, int *p, int rank, int size, comm_lists c);
void exchange_separators(comm_lists c, double *y, int rank, int size);

comm_lists_singleton init_comm_list_singleton(int size);
void free_comm_lists_singleton(comm_lists_singleton *c);
void find_sendlists_singleton(Graph G, int *p, int rank, int size, comm_lists_singleton *c);
void find_recvlists_singleton(Graph G, int *p, int rank, int size, comm_lists_singleton *c);
void exchange_separators_singleton(comm_lists_singleton c, double *y, int rank, int size);

void partition_graph(Graph G, int k, int *p, double *x);

void distribute_graph(Graph *G, int rank);
