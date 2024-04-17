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

// IT: Index Type
// VT: Value Type
template <typename IT, typename VT> class Graph {
  public:
    int N, M, nnz;
    std::vector<IT> vertices, edges;
    std::vector<VT> vals;

    Graph()
        : N(0),
          M(0),
          nnz(0){};

    ~Graph() = default;

    void read_mtx(std::string &file_path, bool is_symmetric = false) {
        std::cout << "Reading mtx file: " << file_path << "\n";
        std::ifstream file(file_path);
        std::vector<IT> r, c;
        std::vector<VT> v;

        fmm::read_options options;
        options.parallel_ok = true;
        options.generalize_symmetry = is_symmetric;

        fmm::read_matrix_market_triplet(file, N, M, r, c, v, options);
        std::cout << "Num rows: " << N << "\n";
        std::cout << "Num cols: " << M << "\n";
        std::cout << "Num non-zeros: " << c.size() << "\n";
        std::cout << "values: ";
        for (auto &i : v)
            std::cout << i << " ";
        std::cout << "\n";

        sort_mtx(r, c, v);
    }

    void sort_mtx(std::vector<IT> &rows, std::vector<IT> &cols, std::vector<VT> &values) {
        std::vector<IT> row_count(N, 0);
        for (int i = 0; i < rows.size(); ++i)
            ++row_count[rows[i]];

        vertices.assign(N + 1, 0);
        std::inclusive_scan(row_count.begin(), row_count.end(), vertices.begin() + 1);
        std::vector<IT> perm(rows.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](IT i, IT j) {
            if (rows[i] != rows[j])
                return rows[i] < rows[j];
            if (cols[i] != cols[j])
                return cols[i] < cols[j];
            return false;
        });

        edges.reserve(cols.size());
        vals.reserve(cols.size());
        std::transform(perm.begin(), perm.end(), std::back_inserter(edges), [&](auto i) { return cols[i]; });
        std::transform(perm.begin(), perm.end(), std::back_inserter(vals), [&](auto i) { return values[i]; });
        nnz = edges.size();
        M = nnz;
        for (int i = 0; i < N; i++) {
            std::cout << i << " -> ";
            for (int row = vertices[i]; row < vertices[i + 1]; row++) {
                std::cout << edges[row] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    /*
     *
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

        for (int i = 0; i < N; i++) {
            std::cout << i << " -> " << partition[i] << std::endl;
        }

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

    void distribute_graph(int rank, int size, std::vector<int> &p) {
        std::cout << "Distributing graph...\n";
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> counts(size, 0);
        std::vector<int> row_displs(p.begin(), p.end() - 1);

        if (rank == 0) {
            for (int i = 0; i < size; i++)
                counts[i] = p[i + 1] - p[i];
        }

        MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < counts.size(); i++) {
                std::cout << "Rank " << i << " gets " << counts[i] << " elements.\n";
            }

            for (auto i : vertices) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
            for (auto i : edges) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        if (rank != 0)
            vertices.assign(counts[rank] + 1, 0);

        MPI_Scatterv(vertices.data(), counts.data(), row_displs.data(), MPI_INT, vertices.data(), counts[rank], MPI_INT,
                     0, MPI_COMM_WORLD);

        // displacement vector for edges
        std::vector<int> col_displs(N, 0);
        if (rank == 0)
            std::inclusive_scan(vertices.begin() + 1, vertices.end(), col_displs.begin() + 1);

        if (rank == 0)
            vertices.resize(p[1] + 1);

        if (rank != 0)
            MPI_Send(&vertices[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        if (rank != size - 1)
            MPI_Recv(&vertices[counts[rank]], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            vertices[counts[rank]] = M;

        MPI_Scatterv(edges.data(), counts.data(), col_displs.data(), MPI_INT, edges.data() + vertices[0], counts[rank],
                     MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 1) {

            for (auto i : edges) {
                std::cout << i << " ";
            }
        }
        std::cout << "Done distributing graph...\n";
    }

    void normalize();
};
