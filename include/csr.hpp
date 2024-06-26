#pragma once
#include <algorithm>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <metis.h>
#include <mpi.h>
#include <numeric>
// #include <omp.h>
#include <set>
#include <vector>

// TODO: Create a struct to store dimensions of matrix, as it is easy to confuse with V and E
template <typename IT, typename VT> class CSR {
  public:
    // V: number of vertices
    // E: number of edges
    // N: Number of rows in matrix
    // M: Number of columns in matrix
    // delta: Offset for row_ptr, i.e. where in the
    int V, E, N, M, offset;
    unsigned long long int nnz;
    std::vector<IT> row_ptr, col_idx;
    std::vector<VT> vals;

    CSR() = default;
    ~CSR() = default;

    /* Partitions a graph into k parts usint METIS_PartGraphRecursive
     * @param k - The number of partitions to make
     * @param p - The partition vector of size |k|+1, where the ith element is the starting index of the ith partition
     * @param A - The input vector for SPMV
     */
    void partition(int k, std::vector<int> &p, std::vector<double> &A) {
        std::cout << "Starting graph partitioning...\n";
        p[0] = 0;

        if (k == 1) {
            p[1] = N;
            return;
        }

        std::vector<int> partition(N, 0);
        int objval, ncon = 1;
        real_t ubvec = 1.01;

        int rc = METIS_PartGraphRecursive(&N, &ncon, row_ptr.data(), col_idx.data(), nullptr, nullptr, nullptr, &k,
                                          nullptr, &ubvec, nullptr, &objval, partition.data());

        std::vector<IT> new_id(N, 0), old_id(N, 0);
        int id = 0;

        for (int r = 0; r < k; r++) {
            for (IT i = 0; i < N; i++) {
                if (partition[i] == r) {
                    old_id[id] = i;
                    new_id[i] = id++;
                }
            }
            p[r + 1] = id;
        }

        std::vector<IT> new_vertices(V + 1, 0);
        std::vector<VT> new_A(N, 0);
        std::vector<IT> new_edges(nnz, 0);
        std::vector<VT> new_vals(nnz, 0);

        for (int i = 0; i < N; i++) {
            int degree = row_ptr[old_id[i] + 1] - row_ptr[old_id[i]];
            new_vertices[i + 1] = new_vertices[i] + degree;

            auto col_start = col_idx.begin() + row_ptr[old_id[i]];
            auto val_start = vals.begin() + row_ptr[old_id[i]];
            std::copy(col_start, col_start + degree, new_edges.begin() + new_vertices[i]);
            std::copy(val_start, val_start + degree, new_vals.begin() + new_vertices[i]);

            for (IT j = new_vertices[i]; j < new_vertices[i + 1]; j++)
                new_edges[j] = new_id[new_edges[j]];
        }

        for (int i = 0; i < N; i++)
            new_A[i] = A[old_id[i]];

        row_ptr = new_vertices;
        col_idx = new_edges;
        vals = new_vals;
        A = new_A;
        std::cout << "Graph partitioning done\n";
    }

    void determine_separators(int k, std::vector<int> &p, std::vector<std::vector<IT>> &separators) {
        std::cout << "Determining separators...\n";
        for (int part = 0; part < k; part++) {
            IT start = p[part], end = p[part + 1];
            std::set<IT> seps;
            for (IT r = start; r < end; r++) {
                std::cout << r << " -> ";
                for (IT idx = row_ptr[r]; idx < row_ptr[r + 1]; idx++) {
                    IT col = col_idx[idx];
                    std::cout << col << " ";
                    if (col < start || col >= end) {
                        seps.insert(col);
                        // break;
                    }
                }
                std::cout << std::endl;
            }
            separators[part].assign(seps.begin(), seps.end());
        }
        std::cout << "Done determining separators...\n";
    }

    // distributes the graph between the processes, ensuring each rank gets their respective partiton of the graph
    void distribute(int rank, int size, std::vector<int> &p, std::vector<VT> &A) {
        if (rank == 0)
            std::cout << "Distributing graph...\n";

        std::vector<IT> vertices_sendcount(size, 0);
        std::vector<IT> edges_displacement(size, 0);
        std::vector<IT> edges_sendcount(size, 0);

        // compute displacement vectors for vertices and edges
        if (rank == 0) {
            std::adjacent_difference(p.begin() + 1, p.end(), vertices_sendcount.begin());
            for (int i = 0; i < size; i++) {
                int start = p[i], end = p[i + 1];
                edges_sendcount[i] = row_ptr[end] - row_ptr[start];
                edges_displacement[i] = row_ptr[start];
            }
        }

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(vertices_sendcount.data(), 1, MPI_INT, &V, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(edges_sendcount.data(), 1, MPI_INT, &E, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            row_ptr.assign(V + 1, 0);
            A.assign(N, 0);
            col_idx.assign(E, 0);
            vals.assign(E, 0);
        }

        MPI_Scatterv(row_ptr.data(), vertices_sendcount.data(), p.data(), MPI_INT, row_ptr.data(), V, MPI_INT, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(col_idx.data(), edges_sendcount.data(), edges_displacement.data(), MPI_INT, col_idx.data(), E,
                     MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vals.data(), edges_sendcount.data(), edges_displacement.data(), MPI_DOUBLE, vals.data(), E,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(A.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            row_ptr.resize(p[1] + 1);
            col_idx.resize(row_ptr[p[1]]);
            vals.resize(row_ptr[p[1]]);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // communicate overlap
        if (rank != 0)
            MPI_Send(&row_ptr[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        if (rank != size - 1)
            MPI_Recv(&row_ptr[V], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            row_ptr[V] = nnz;

        offset = row_ptr[0];
        for (auto &i : row_ptr)
            i -= offset;

        if (rank == 0)
            std::cout << "Done distributing graph..." << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    void normalize();
};
