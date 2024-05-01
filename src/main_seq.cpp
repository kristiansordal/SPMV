#include "spmv.hpp"
#include <csr.hpp>
#include <mtx.hpp>
#include <omp.h>

int main(int argc, char **argv) {
    MTX<int, double> mtx;
    CSR<int, double> g;
    std::string file = argv[1];
    mtx.read_mtx(file);
    g = mtx.mtx_to_csr();
    std::vector<double> A(g.N, 0), y(g.N, 0);
    std::cout << A.size() << " " << y.size() << std::endl;

    int num_steps = 100;
    double start, end;
    unsigned long long int ops;

    for (int i = 0; i < g.N; i++)
        A[i] = ((double)rand() / (RAND_MAX)) + 1;

    start = omp_get_wtime();
    for (int i = 0; i < num_steps; i++) {
        spmv(g, A, y);
        std::swap(A, y);
    }

    ops = 2 * g.nnz * num_steps;
    std::cout << ops << std::endl;
    end = omp_get_wtime();

    std::cout << "Time: " << end - start << "\nGFLOPS: " << ops / ((end - start) * 1e9) << "\n";
}
