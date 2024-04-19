#pragma once
#include <algorithm>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <metis.h>
#include <mpi.h>
#include <numeric>
#include <set>
#include <vector>
namespace fmm = fast_matrix_market;

// TODO: refactor variable names to something more sensible. Right now there is a mixture of using both "rows"
// "vertices" "cols" and "edges"
// IT: Index Type
// VT: Value Type
template <typename IT, typename VT> class Graph {
  public:
    int N, M, nnz;
    int n_local_vertices, n_local_edges;
    std::vector<IT> vertices, edges;
    std::vector<VT> vals;

    Graph()
        : N(0),
          M(0),
          nnz(0){};
    ~Graph() = default;

    void read_mtx(std::string &file_path /* , bool is_symmetric = false */) {
        std::cout << "Reading mtx file: " << file_path << "\n";
        std::ifstream file(file_path);
        std::vector<IT> r, c;
        std::vector<VT> v;

        fmm::read_options options;
        options.parallel_ok = true;
        // options.generalize_symmetry = is_symmetric;

        fmm::read_matrix_market_triplet(file, N, M, r, c, v, options);
        std::cout << "|V| = " << N << " |E| = " << M << "\n";
        sort_mtx(r, c, v);
    }

    // void sort_mtx(std::vector<IT> &rows, std::vector<IT> &cols, std::vector<VT> &values) {
    //     std::cout << "Sorting mtx file...\n";
    //     std::vector<IT> row_count(N + 1, 0);

    //     for (auto &v : rows)
    //         ++row_count[v];
    //     std::cout << "Got row count" << std::endl;

    //     vertices.assign(N + 1, 0);
    //     std::exclusive_scan(row_count.begin(), row_count.end(), vertices.begin(), 0);
    //     std::vector<IT> perm(rows.size());
    //     std::iota(perm.begin(), perm.end(), 0);
    //     std::cout << "iota,ex scan, vert assign done" << std::endl;
    //     std::sort(perm.begin(), perm.end(), [rows, cols](IT i, IT j) {
    //         if (rows[i] != rows[j])
    //             return rows[i] < rows[j];
    //         if (cols[i] != cols[j])
    //             return cols[i] < cols[j];
    //         return false;
    //     });
    //     std::cout << "sorting done" << std::endl;

    //     edges.reserve(cols.size());
    //     vals.reserve(cols.size());
    //     std::cout << "edges, vals reserve done" << std::endl;
    //     std::transform(perm.begin(), perm.end(), std::back_inserter(edges), [&](auto i) { return cols[i]; });
    //     std::transform(perm.begin(), perm.end(), std::back_inserter(vals), [&](auto i) { return values[i]; });
    //     std::cout << "transform done" << std::endl;
    //     nnz = edges.size();
    //     M = nnz;
    //     std::cout << "Done sorting mtx file...\n";
    // }
    void sort_mtx(std::vector<IT> &rows, std::vector<IT> &cols, std::vector<VT> &values) {
        std::cout << "Sorting mtx file..." << std::endl;
        std::vector<std::tuple<IT, IT, VT>> tuples;
        tuples.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            tuples.emplace_back(rows[i], cols[i], values[i]);

        std::cout << "Start std::sort" << std::endl;
        std::sort(tuples.begin(), tuples.end(), [](const auto &a, const auto &b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
        });
        std::cout << "End std::sort" << std::endl;

        for (size_t i = 0; i < tuples.size(); ++i) {
            rows[i] = std::get<0>(tuples[i]);
            cols[i] = std::get<1>(tuples[i]);
            values[i] = std::get<2>(tuples[i]);
        }
        std::cout << "Done sorting mtx file..." << std::endl;
    }

    /* Partitions a graph into k parts usint METIS_PartGraphRecursive
     *
     * @param k - The number of partitions to make
     * @param p - The partition vector of size |k|+1, where the ith element is the starting index of the ith partition
     * @param v_old - The input vector for SPMV
     */
    void partition_graph(int k, std::vector<int> &p, std::vector<double> &v_old) {
        std::cout << "Starting graph partitioning...\n";
        p[0] = 0;

        if (k == 1) {
            p[1] = N;
            return;
        }

        std::vector<int> partition(N, 0);
        int objval, ncon = 1;
        real_t ubvec = 1.01;

        int rc = METIS_PartGraphRecursive(&N, &ncon, vertices.data(), edges.data(), nullptr, nullptr, nullptr, &k,
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

        std::vector<IT> new_vertices(N + 1, 0);
        std::vector<IT> new_edges(M, 0);
        std::vector<VT> new_vals(M, 0);

        for (int i = 0; i < N; i++) {
            int degree = vertices[old_id[i] + 1] - vertices[old_id[i]];
            new_vertices[i + 1] = new_vertices[i] + degree;

            auto col_start = edges.begin() + vertices[old_id[i]];
            auto val_start = vals.begin() + vertices[old_id[i]];
            std::copy(col_start, col_start + degree, new_edges.begin() + new_vertices[i]);
            std::copy(val_start, val_start + degree, new_vals.begin() + new_vertices[i]);

            for (IT j = new_vertices[i]; j < new_vertices[i + 1]; j++)
                new_edges[j] = new_id[new_edges[j]];
        }

        std::vector<VT> new_v(N, 0);
        for (int i = 0; i < N; i++)
            new_v[i] = v_old[old_id[i]];

        vertices = new_vertices;
        edges = new_edges;
        vals = new_vals;
        v_old = new_v;
        std::cout << "Graph partitioning done\n";
    }

    void determine_separators(int k, std::vector<int> &p, std::vector<std::vector<IT>> &separators) {
        std::cout << "Determining separators...\n";
        for (int part = 0; part < k; part++) {
            IT start = p[part], end = p[part + 1];
            std::set<IT> seps;
            for (IT r = start; r < end; r++) {
                for (IT idx = vertices[r]; idx < vertices[r + 1]; idx++) {
                    IT col = edges[idx];
                    if (col < start || col >= end) {
                        seps.insert(col);
                        std::cout << "Separator: " << col << std::endl;
                    }
                }
            }
            separators[part].assign(seps.begin(), seps.end());
        }
        std::cout << "Done determining separators...\n";
    }

    // distributes the graph between the processes, ensuring each rank gets their respective partiton of the graph
    void distribute_graph(int rank, int size, std::vector<int> &p) {
        std::cout << "Distributing graph...\n";

        std::vector<IT> vertices_displacement(p.begin(), p.end() - 1);
        std::vector<IT> vertices_sendcount(size, 0);
        std::vector<IT> edges_displacement(size, 0);
        std::vector<IT> edges_sendcount(size, 0);

        // compute displacement vectors for vertices and edges
        if (rank == 0) {
            std::adjacent_difference(p.begin() + 1, p.end(), vertices_sendcount.begin());

            for (int i = 0; i < size; i++) {
                int start = p[i], end = p[i + 1];
                edges_sendcount[i] = vertices[end] - vertices[start];
                edges_displacement[i] = vertices[start];
            }
        }

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(vertices_sendcount.data(), 1, MPI_INT, &n_local_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(edges_sendcount.data(), 1, MPI_INT, &n_local_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            vertices.assign(n_local_vertices + 1, 0);
            edges.assign(n_local_edges, 0);
            vals.assign(n_local_edges, 0);
        }

        MPI_Scatterv(vertices.data(), vertices_sendcount.data(), vertices_displacement.data(), MPI_INT, vertices.data(),
                     n_local_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(edges.data(), edges_sendcount.data(), edges_displacement.data(), MPI_INT, edges.data(),
                     n_local_edges, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
            vertices.resize(p[1] + 1);

        // communicate overlap
        if (rank != 0)
            MPI_Send(&vertices[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        if (rank != size - 1)
            MPI_Recv(&vertices[vertices_sendcount[rank]], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            vertices[vertices_sendcount[rank]] = N;
    }

    void normalize();
};
