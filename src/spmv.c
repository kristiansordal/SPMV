#include "spmv.h"
#include <metis.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void spmv(CSR g, double *x, double *y, long long int *flops) {
    for (int u = 0; u < g.num_rows; u++) {
        double z = 0.0;
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            int v = g.col_idx[i];
            z += x[v] * g.values[i];
        }
        y[u] = z;
    }
}

void spmv_part(CSR g, int rank, int row_ptr_start_idx, int row_ptr_end_idx, double *x, double *y) {
#pragma omp parallel for schedule(static)
    for (int u = row_ptr_start_idx; u < row_ptr_end_idx; u++) {
        double z = 0.0;
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            int v = g.col_idx[i];
            z += x[v] * g.values[i];
        }
        y[u] = z;
    }
}

void partition_graph(CSR g, int num_partitions, int *partition_idx) {
    if (num_partitions == 1) {
        partition_idx[0] = 0;
        partition_idx[1] = g.num_rows;
        return;
    }

    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.num_rows);
    int rc = METIS_PartGraphKway(&g.num_rows, &ncon, g.row_ptr, g.col_idx, NULL, NULL, NULL, &num_partitions, NULL,
                                 &ubvec, NULL, &objval, part);

    // new_id[i] stores the new position of node i
    // old_id[i] stores the old position of node i
    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);
    int id = 0;
    partition_idx[0] = 0;
    for (int r = 0; r < num_partitions; r++) {
        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        partition_idx[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_E = malloc(sizeof(int) * g.num_cols);
    double *new_A = malloc(sizeof(double) * g.num_cols);

    new_V[0] = 0;
    for (int i = 0; i < g.num_rows; i++) {
        int d = g.row_ptr[old_id[i] + 1] - g.row_ptr[old_id[i]];
        new_V[i + 1] = new_V[i] + d;
        memcpy(new_E + new_V[i], g.col_idx + g.row_ptr[old_id[i]], sizeof(int) * d);
        memcpy(new_A + new_V[i], g.values + g.row_ptr[old_id[i]], sizeof(double) * d);

        for (int j = new_V[i]; j < new_V[i + 1]; j++) {
            new_E[j] = new_id[new_E[j]];
        }
    }

    memcpy(g.row_ptr, new_V, sizeof(int) * (g.num_rows + 1));
    memcpy(g.col_idx, new_E, sizeof(int) * g.num_cols);
    memcpy(g.values, new_A, sizeof(double) * g.num_cols);

    free(new_V);
    free(new_E);
    free(new_A);

    free(new_id);
    free(old_id);
    free(part);
}

void partition_graph_1b(CSR g, int num_partitions, int *partition_idx, comm_lists *c) {
    if (num_partitions == 1) {
        partition_idx[0] = 0;
        partition_idx[1] = g.num_rows;
        return;
    }

    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.num_rows);
    int rc = METIS_PartGraphKway(&g.num_rows, &ncon, g.row_ptr, g.col_idx, NULL, NULL, NULL, &num_partitions, NULL,
                                 &ubvec, NULL, &objval, part);

    int *sep_marker = malloc(g.num_rows * sizeof(int));
    for (int i = 0; i < num_partitions; i++)
        c->send_count[i] = 0;
    for (int i = 0; i < g.num_rows; i++)
        sep_marker[i] = 0;

    int sep = 0;
    for (int i = 0; i < g.num_rows; i++) {
        for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
            if (part[i] != part[g.col_idx[j]]) {
                sep_marker[i] = 1;
                c->send_count[part[i]]++;
                break;
            }
        }
    }

    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);
    int id = 0;
    partition_idx[0] = 0;
    for (int r = 0; r < num_partitions; r++) {
        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }

        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && !sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }

        partition_idx[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_E = malloc(sizeof(int) * g.num_cols);
    double *new_A = malloc(sizeof(double) * g.num_cols);

    new_V[0] = 0;
    for (int i = 0; i < g.num_rows; i++) {
        int d = g.row_ptr[old_id[i] + 1] - g.row_ptr[old_id[i]];
        new_V[i + 1] = new_V[i] + d;
        memcpy(new_E + new_V[i], g.col_idx + g.row_ptr[old_id[i]], sizeof(int) * d);
        memcpy(new_A + new_V[i], g.values + g.row_ptr[old_id[i]], sizeof(double) * d);

        for (int j = new_V[i]; j < new_V[i + 1]; j++) {
            new_E[j] = new_id[new_E[j]];
        }
    }

    memcpy(g.row_ptr, new_V, sizeof(int) * (g.num_rows + 1));
    memcpy(g.col_idx, new_E, sizeof(int) * g.num_cols);
    memcpy(g.values, new_A, sizeof(double) * g.num_cols);

    free(new_V);
    free(new_E);
    free(new_A);

    free(new_id);
    free(old_id);
    free(part);
}

void partition_graph_1c(CSR g, int num_partitions, int *partition_idx, comm_lists *c) {
    if (num_partitions == 1) {
        partition_idx[0] = 0;
        partition_idx[1] = g.num_rows;
        return;
    }

    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.num_rows);
    int rc = METIS_PartGraphKway(&g.num_rows, &ncon, g.row_ptr, g.col_idx, NULL, NULL, NULL, &num_partitions, NULL,
                                 &ubvec, NULL, &objval, part);

    int *sep_marker = malloc(g.num_rows * sizeof(int));
    for (int i = 0; i < num_partitions; i++)
        c->send_count[i] = 0;
    for (int i = 0; i < g.num_rows; i++)
        sep_marker[i] = 0;

    int sep = 0;

    for (int i = 0; i < g.num_rows; i++) {
        for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
            if (part[i] != part[g.col_idx[j]]) {
                sep_marker[i] = 1;
                c->send_count[part[i]]++;
                c->send_items[part[i]][part[g.col_idx[j]]] = 1;
                c->send_items[part[g.col_idx[j]]][part[i]] = 1;
                break;
            }
        }
    }

    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);
    int id = 0;
    partition_idx[0] = 0;
    for (int r = 0; r < num_partitions; r++) {
        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }

        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && !sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }

        partition_idx[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_E = malloc(sizeof(int) * g.num_cols);
    double *new_A = malloc(sizeof(double) * g.num_cols);

    new_V[0] = 0;
    for (int i = 0; i < g.num_rows; i++) {
        int d = g.row_ptr[old_id[i] + 1] - g.row_ptr[old_id[i]];
        new_V[i + 1] = new_V[i] + d;
        memcpy(new_E + new_V[i], g.col_idx + g.row_ptr[old_id[i]], sizeof(int) * d);
        memcpy(new_A + new_V[i], g.values + g.row_ptr[old_id[i]], sizeof(double) * d);

        for (int j = new_V[i]; j < new_V[i + 1]; j++) {
            new_E[j] = new_id[new_E[j]];
        }
    }

    memcpy(g.row_ptr, new_V, sizeof(int) * (g.num_rows + 1));
    memcpy(g.col_idx, new_E, sizeof(int) * g.num_cols);
    memcpy(g.values, new_A, sizeof(double) * g.num_cols);

    free(new_V);
    free(new_E);
    free(new_A);

    free(new_id);
    free(old_id);
    free(part);
}

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c) {
    int *send_mark = malloc(sizeof(int) * g.num_rows);
    for (int r = 0; r < size; r++) {
        c.send_count[r] = 0;
        c.send_items[r] = NULL;
        c.send_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[rank]; i < p[rank + 1]; i++)
            send_mark[i] = 0;

        // Find separators
        for (int u = p[r]; u < p[r + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[rank] && v < p[rank + 1])
                    send_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[rank]; i < p[rank + 1]; i++)
            c.send_count[r] += send_mark[i];

        if (c.send_count[r] == 0)
            continue;

        // Store list of separators
        c.send_items[r] = malloc(sizeof(int) * c.send_count[r]);
        c.send_lists[r] = malloc(sizeof(double) * c.send_count[r]);

        int j = 0;
        for (int i = p[rank]; i < p[rank + 1]; i++) {
            if (send_mark[i]) {
                c.send_items[r][j++] = i;
            }
        }
    }

    free(send_mark);
}

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c) {
    int *receive_mark = malloc(sizeof(int) * g.num_rows);
    for (int r = 0; r < size; r++) {
        c.receive_count[r] = 0;
        c.receive_items[r] = NULL;
        c.receive_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[r]; i < p[r + 1]; i++)
            receive_mark[i] = 0;

        // Find separators
        for (int u = p[rank]; u < p[rank + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[r] && v < p[r + 1])
                    receive_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[r]; i < p[r + 1]; i++)
            c.receive_count[r] += receive_mark[i];

        if (c.receive_count[r] == 0)
            continue;

        // Store list of separators
        c.receive_items[r] = malloc(sizeof(int) * c.receive_count[r]);
        c.receive_lists[r] = malloc(sizeof(double) * c.receive_count[r]);

        int j = 0;
        for (int i = p[r]; i < p[r + 1]; i++)
            if (receive_mark[i])
                c.receive_items[r][j++] = i;
    }

    free(receive_mark);
}

// attempts to make good load balancing without splitting the rows.
void partition_graph_naive(CSR g, int s, int t, int k, int *p) {
    int edges_per = (g.row_ptr[t] - g.row_ptr[s]) / k;
    p[0] = s;
    int id = 1;
    for (int u = s; u < t; u++) {
        if ((g.row_ptr[u] - g.row_ptr[s]) >= edges_per * id)
            p[id++] = u;
    }
    while (id <= k)
        p[id++] = t;
    p[k] = t;
}

void distribute_graph(CSR *g, int rank) {
    MPI_Bcast(&g->num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g->num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        g->row_ptr = malloc(sizeof(int) * (g->num_rows + 1));
        g->col_idx = malloc(sizeof(int) * g->num_cols);
        g->values = malloc(sizeof(double) * g->num_cols);
    }

    MPI_Bcast(g->row_ptr, g->num_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->col_idx, g->num_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->values, g->num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

comm_lists init_comm_lists(int size) {
    comm_lists c = {.send_count = malloc(sizeof(int) * size),
                    .receive_count = malloc(sizeof(int) * size),
                    .send_items = malloc(sizeof(int *) * size),
                    .receive_items = malloc(sizeof(int *) * size),
                    .send_lists = malloc(sizeof(double *) * size),
                    .receive_lists = malloc(sizeof(double *) * size)};
    return c;
}

void free_comm_lists(comm_lists *c, int size) {
    for (int i = 0; i < size; i++) {
        free(c->send_items[i]);
        free(c->send_lists[i]);
        free(c->receive_items[i]);
        free(c->receive_lists[i]);
    }

    free(c->send_count);
    free(c->receive_count);

    free(c->send_items);
    free(c->receive_items);

    free(c->send_lists);
    free(c->receive_lists);
}

void exchange_separators(comm_lists c, double *y, int *displs, int rank, int size) {
    MPI_Request *requests = malloc(sizeof(MPI_Request) * 2 * size);
    int req_count = 0;

    for (int r = 0; r < size; r++) {
        if (rank == r || c.send_items[r][rank] == 0) // If r doesn't send to me
            continue;
        MPI_Irecv(y + displs[r], c.send_count[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    for (int r = 0; r < size; r++) {
        if (rank == r || c.send_items[rank][r] == 0) // If I don't send to r
            continue;
        MPI_Isend(y + displs[rank], c.send_count[rank], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    free(requests);
}

void exchange_required_separators(comm_lists c, double *Vn, int rank, int size) {
    int total_send = 0, total_recv = 0;

    // Compute total send and receive counts
    for (int i = 0; i < size; i++) {
        total_send += c.send_count[i];
        total_recv += c.receive_count[i];
    }

    // Allocate send and receive buffers
    double *send_buffer = malloc(sizeof(double) * total_send);
    double *recv_buffer = malloc(sizeof(double) * total_recv);

    // Pack send buffer
    int send_offset = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < c.send_count[i]; j++) {
            send_buffer[send_offset++] = Vn[c.send_items[i][j]];
        }
    }

    // Compute displacement arrays for MPI_Ialltoallv
    int *sdispls = malloc(sizeof(int) * size);
    int *rdispls = malloc(sizeof(int) * size);
    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 1; i < size; i++) {
        sdispls[i] = sdispls[i - 1] + c.send_count[i - 1];
        rdispls[i] = rdispls[i - 1] + c.receive_count[i - 1];
    }

    MPI_Request *reqs = malloc(sizeof(MPI_Request) * size);
    int num_reqs = 0;
    MPI_Ialltoallv(send_buffer, c.send_count, sdispls, MPI_DOUBLE, recv_buffer, c.receive_count, rdispls, MPI_DOUBLE,
                   MPI_COMM_WORLD, &reqs[num_reqs++]);

    MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

    // Unpack received values into Vn
    int recv_offset = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < c.receive_count[i]; j++) {
            Vn[c.receive_items[i][j]] = recv_buffer[recv_offset++];
        }
    }

    free(send_buffer);
    free(recv_buffer);
    free(sdispls);
    free(rdispls);
}
