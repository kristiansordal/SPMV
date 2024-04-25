#pragma once
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <graph.hpp>
#include <numeric>
#include <tuple>
#include <vector>
namespace fmm = fast_matrix_market;
template <typename IT, typename VT> class MTX {
  private:
    int M, N, nnz;
    std::vector<IT> rows, cols;
    std::vector<VT> vals;
    std::vector<std::tuple<IT, IT, VT>> triplets;

  public:
    MTX() = default;
    ~MTX() = default;

    /* Reads a .mtx file and stores the values in the triplet vector
     * @param file_path: path to the .mtx file
     */
    void read_mtx(const std::string &file_path) {
        std::cout << "Reading mtx file: " << file_path << "\n";
        std::ifstream file(file_path);

        fmm::read_options options;
        options.parallel_ok = true;

        fmm::read_matrix_market_triplet(file, N, M, rows, cols, vals, options);
        nnz = rows.size();
        triplets.resize(nnz);
        for (int i = 0; i < nnz; i++)
            triplets[i] = {rows[i], cols[i], vals[i]};

        rows.clear();
        cols.clear();
        vals.clear();
    }

    /* Converts the triplet vector to CSR format
     * @return: Graph object in CSR format
     */
    Graph<IT, VT> mtx_to_csr() {
        std::cout << "Converting from MTX to CSR...\n";
        Graph<IT, VT> graph{};
        std::vector<IT> row_count(N + 1, 0);
        graph.row_ptr.resize(N + 1);
        graph.col_idx.resize(nnz);
        graph.vals.resize(nnz);

        std::sort(triplets.begin(), triplets.end(), [](const auto &a, const auto &b) {
            if (std::get<0>(a) == std::get<0>(b))
                return std::get<1>(a) < std::get<1>(b);
            return std::get<0>(a) < std::get<0>(b);
        });

        for (int i = 0; i < nnz; i++) {
            auto triplet = triplets[i];
            row_count[std::get<0>(triplet)]++;
            graph.col_idx[i] = std::get<1>(triplet);
            graph.vals[i] = std::get<2>(triplet);
        }

        std::exclusive_scan(row_count.begin(), row_count.end(), graph.row_ptr.begin(), 0);
        graph.N = N;
        graph.M = M;
        graph.V = N;
        graph.E = nnz;
        graph.nnz = nnz;
        graph.row_ptr[N] = nnz;
        std::cout << "Done converting from MTX to CSR...\n";
        return graph;
    }
};
