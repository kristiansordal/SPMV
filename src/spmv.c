#include <metis.h>
#include <mpi.h>
#include <spmv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

comm_lists init_comm_lists(int size) {
    comm_lists c = {.send_count = malloc(sizeof(int) * size),
                    .recv_count = malloc(sizeof(int) * size),
                    .send_items = malloc(sizeof(int *) * size),
                    .recv_items = malloc(sizeof(int *) * size),
                    .send_lists = malloc(sizeof(double *) * size),
                    .recv_lists = malloc(sizeof(double *) * size)};
    return c;
}
comm_lists_singleton init_comm_list_singleton(int size) {
    comm_lists_singleton c = {.send_count = 0,
                              .recv_count = 0,
                              .send_items = malloc(sizeof(int *) * size),
                              .recv_items = malloc(sizeof(int *) * size),
                              .send_list = malloc(sizeof(double *) * size),
                              .recv_list = malloc(sizeof(double *) * size)};
    return c;
}

void free_comm_lists(comm_lists *c, int size) {
    for (int i = 0; i < size; i++) {
        free(c->send_items[i]);
        free(c->send_lists[i]);
        free(c->recv_items[i]);
        free(c->recv_lists[i]);
    }

    free(c->send_count);
    free(c->recv_count);

    free(c->send_items);
    free(c->recv_items);

    free(c->send_lists);
    free(c->recv_lists);
}

void free_comm_lists_singleton(comm_lists_singleton *c) {
    free(c->send_items);
    free(c->recv_items);

    free(c->send_list);
    free(c->recv_list);
}

/**
 * @brief Find the send lists for each rank
 *
 * @param G - The graph
 * @param p - The partition array
 * @param rank - The rank
 * @param size - The number of ranks
 * @param c - The communication lists
 */
void find_sendlists(Graph G, int *p, int rank, int size, comm_lists c) { return; }

void find_sendlists_singleton(Graph G, int *p, int rank, int size, comm_lists_singleton *c) {
    int *send_mark = malloc(sizeof(int) * G.n);
    memset(send_mark, 0, sizeof(int) * G.n); // dont know what to send yet, set to zero
    c->send_count = 0;
    int ptr = 0;

    // for each rank, check if they need any of the values computed by my rank
    // i.e. they need an entry in the rank p[rank] -> p[rank+1]
    for (int r = 0; r < size; r++) {
        // dont send to self
        if (r == rank)
            continue;

        for (int v = p[rank]; v < p[rank + 1]; v++) {
            for (int u = G.vertices[v]; u < G.vertices[v + 1]; u++) {
                int nbr = G.edges[u];
                if (nbr >= p[r] && nbr < p[r + 1])
                    send_mark[nbr] = 1;
            }
        }
    }

    for (int i = 0; i < G.n; i++)
        c->send_count += send_mark[i];

    c->send_list = malloc(sizeof(double) * c->send_count);
    c->send_items = malloc(sizeof(int) * c->send_count);

    for (int i = 0; i < G.n; i++) {
        if (send_mark[i]) {
            c->send_list[ptr] = G.vals[i];
            c->send_items[ptr++] = i;
        }
    }

    free(send_mark);
}

/**
 * @brief Partition the graph G into k parts
 *
 * @param G - The graph
 * @param k - The number of parts (equal to number of ranks)
 * @param p - The partition array
 * @param x - The vector
 */
void partition_graph(Graph G, int k, int *p, double *v_old) {
    if (k == 1) {
        p[0] = 0;
        p[1] = G.n;
        return;
    }

    int objval, ncon = 1; // Number of balancing constraints
    real_t ubvec = 1.01;  // Unbalance tolerance - 1.01 allows for 1% imbalance

    // Store the result of the partiton. This array will contain values in the rank 0 < x < k
    // Value at index i tells us to whick part x the index belongs to.
    int *partition = malloc(sizeof(int) * G.n);

    int rc = METIS_PartGraphKway(&G.n, &ncon, G.vertices, G.edges, NULL, NULL, NULL, &k, NULL, &ubvec, NULL, &objval,
                                 partition);

    int *new_id = malloc(sizeof(int) * G.n);
    int *old_id = malloc(sizeof(int) * G.n);
    int id = 0;
    p[0] = 0;

    // Group vertices by their partition and assign new IDs
    for (int r = 0; r < k; r++) {
        for (int i = 0; i < G.n; i++) {
            if (partition[i] == r) {
                old_id[id] = i;   // Record original vertex ID
                new_id[i] = id++; // Assign new ID and increment counter
            }
        }
        p[r + 1] = id; // Record the end index of the current partition
    }

    // Allocate memory for reordering the graph's vertices and edges
    int *new_V = malloc(sizeof(int) * (G.n + 1));
    int *new_E = malloc(sizeof(int) * G.m);
    double *new_A = malloc(sizeof(double) * G.m); // New values array

    // Reorder vertices and their edges according to the new partitioning
    new_V[0] = 0;
    for (int i = 0; i < G.n; i++) {
        int degree = G.vertices[old_id[i] + 1] - G.vertices[old_id[i]]; // Degree of the current vertex
        new_V[i + 1] = new_V[i] + degree; // Update the vertex index for the reordered graph
        memcpy(new_E + new_V[i], G.edges + G.vertices[old_id[i]], sizeof(int) * degree);
        memcpy(new_A + new_V[i], G.vals + G.vertices[old_id[i]], sizeof(double) * degree);

        // Update edge targets to new IDs
        for (int j = new_V[i]; j < new_V[i + 1]; j++)
            new_E[j] = new_id[new_E[j]];
    }

    // Reorder the vector x according to the new partitioning
    double *new_X = malloc(sizeof(double) * G.n);
    for (int i = 0; i < G.n; i++) {
        new_X[i] = v_old[old_id[i]];
    }
    memcpy(v_old, new_X, sizeof(double) * G.n);

    memcpy(G.vertices, new_V, sizeof(int) * (G.n + 1));
    memcpy(G.edges, new_E, sizeof(int) * G.m);
    memcpy(G.vals, new_A, sizeof(double) * G.m);

    free(new_V);
    free(new_E);
    free(new_A);
    free(new_X);

    free(new_id);
    free(old_id);
    free(partition);
}

void distribute_graph(Graph *G, int rank) {
    MPI_Bcast(&G->n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&G->m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        G->vertices = malloc(sizeof(int) * (G->n + 1));
        G->edges = malloc(sizeof(int) * G->m);
        G->vals = malloc(sizeof(double) * G->m);
    }

    MPI_Bcast(G->vertices, G->n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(G->edges, G->m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(G->vals, G->m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
