#include <metis.h>
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

void find_sendlists(Graph G, int *p, int rank, int size, comm_lists c) { return; }

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
    int *part = malloc(sizeof(int) * G.n);

    int rc =
        METIS_PartGraphKway(&G.n, &ncon, G.vertices, G.edges, NULL, NULL, NULL, &k, NULL, &ubvec, NULL, &objval, part);

    int *new_id = malloc(sizeof(int) * G.n);
    int *old_id = malloc(sizeof(int) * G.n);
    int id = 0;
    p[0] = 0;

    // Group vertices by their partition and assign new IDs
    for (int r = 0; r < k; r++) {
        for (int i = 0; i < G.n; i++) {
            if (part[i] == r) {
                old_id[id] = i;   // Record original vertex ID
                new_id[i] = id++; // Assign new ID and increment counter
            }
        }
        p[r + 1] = id; // Record the end index of the current partition
    }

    printf("HERE BRP\n");
    // Allocate memory for reordering the graph's vertices and edges
    int *new_vertices = malloc(sizeof(int) * (G.n * 2));
    int *new_edges = malloc(sizeof(int) * G.m * 2);
    double *new_vals = malloc(sizeof(double) * G.m * 2); // New values array

    // Reorder vertices and their edges according to the new partitioning
    new_vertices[0] = 0;
    for (int i = 0; i < G.n; i++) {
        int degree = G.vertices[old_id[i] + 1] - G.vertices[old_id[i]]; // Degree of the current vertex
        new_vertices[i + 1] = new_vertices[i] + degree; // Update the vertex index for the reordered graph
        printf("Copying edges for vertex %d from original index %d to new index %d\n", i, G.vertices[old_id[i]],
               new_vertices[i]);
        printf("Number of edges (degree): %d\n", degree);
        memcpy(new_edges + new_vertices[i], G.edges + G.vertices[old_id[i]], sizeof(int) * degree);  // Reorder edges
        memcpy(new_vals + new_vertices[i], G.vals + G.vertices[old_id[i]], sizeof(double) * degree); // Reorder values

        // Update edge targets to new IDs
        for (int j = new_vertices[i]; j < new_vertices[i + 1]; j++) {
            new_edges[j] = new_id[new_edges[j]];
        }
    }
    printf("done\n");

    // Reorder the vector x according to the new partitioning
    double *new_X = malloc(sizeof(double) * G.n);
    for (int i = 0; i < G.n; i++) {
        new_X[i] = v_old[old_id[i]];
    }
    memcpy(v_old, new_X, sizeof(double) * G.n);

    memcpy(v_old, new_X, sizeof(double) * G.n);

    memcpy(G.vertices, new_vertices, sizeof(int) * (G.n + 1));
    memcpy(G.edges, new_edges, sizeof(int) * G.m);
    memcpy(G.vals, new_vals, sizeof(double) * G.m);

    free(new_vertices);
    free(new_edges);
    free(new_vals);
    free(new_X);

    free(new_id);
    free(old_id);
    free(part);
}
