#pragma once
#include <csr.hpp>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <tuple>
#include <vector>
namespace fmm = fast_matrix_market;
template <typename IT, typename VT> class MTX {
  private:
    int N, M, nnz;
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
        std::cout << "Reading MTX file: " << file_path << "\n";
        std::ifstream file(file_path);

        fmm::read_options options;
        options.parallel_ok = true;

        fmm::read_matrix_market_triplet(file, N, M, rows, cols, vals, options);
        nnz = rows.size();
        triplets.resize(nnz);

#pragma omp parallel for
        for (int i = 0; i < nnz; i++)
            triplets[i] = {rows[i], cols[i], vals[i]};

        std::cout << "|V| = " << N << " |E| = " << nnz << "\n";
        std::cout << "Done reading MTX file...\n";
    }

    double l2_norm_triplet() {
        double norm = 0;
        for (int i = 0; i < nnz; i++)
            norm += std::get<2>(triplets[i]) * std::get<2>(triplets[i]);
        return sqrt(norm);
    }

    /* Converts the triplet vector to CSR format
     * @return: Graph object in CSR format
     */
    CSR<IT, VT> mtx_to_csr() {
        std::cout << "Converting from MTX to CSR...\n";
        CSR<IT, VT> graph{};
        std::vector<IT> row_count(N + 1, 0);
        graph.row_ptr.resize(N + 1);
        graph.col_idx.resize(nnz);
        graph.vals.resize(nnz);

        std::sort(triplets.begin(), triplets.end(), [](const auto &a, const auto &b) {
            if (std::get<0>(a) == std::get<0>(b))
                return std::get<1>(a) < std::get<1>(b);
            return std::get<0>(a) < std::get<0>(b);
        });

#pragma omp parallel for
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
        std::cout << "Done converting from MTX to CSR...\n";
        return graph;
    }
};
