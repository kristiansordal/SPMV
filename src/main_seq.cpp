#include "spmv.hpp"
#include <graph.hpp>
#include <mtx.hpp>
#include <omp.h>

int main(int argc, char **argv) {
    MTX<int, double> mtx;
    Graph<int, double> g;
    std::string file = argv[1];
    mtx.read_mtx(file);
    g = mtx.mtx_to_csr();
    std::vector<double> A(g.N, 0), y(g.N, 0);
    std::cout << A.size() << " " << y.size() << std::endl;

    auto num_steps = 100;
    double start, end;
    unsigned long long ops;

    for (int i = 0; i < g.N; i++)
        A[i] = ((double)rand() / (RAND_MAX)) + 1;

    start = omp_get_wtime();
    for (int i = 0; i < num_steps; i++) {
        spmv(g, A, y);
        std::swap(A, y);
    }

    unsigned long long max_value = std::numeric_limits<unsigned long long>::max();
    if (num_steps > 0 && max_value / num_steps < 2 * g.nnz) {
        std::cerr << "Overflow will occur!\n";
    } else {
        ops = 2 * g.nnz * num_steps;
    }
    std::cout << g.nnz << " " << num_steps << std::endl;
    std::cout << g.nnz * num_steps << std::endl;
    std::cout << 2 * g.nnz * num_steps << std::endl;
    ops = 2 * g.nnz * num_steps;
    std::cout << ops << std::endl;
    end = omp_get_wtime();

    std::cout << "Time: " << end - start << "\nGFLOPS: " << ops / ((end - start) * 1e9) << "\n";
}
