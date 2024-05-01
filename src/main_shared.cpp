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

    int num_steps = 100;
    double t_start, t_end;
    unsigned long long int ops;

    for (int i = 0; i < g.N; i++)
        A[i] = ((double)rand() / (RAND_MAX)) + 1;

    t_start = omp_get_wtime();
    while (num_steps--) {
        spmv_shared(g, A, y);
        std::swap(A, y);
    }

    ops = 2 * g.nnz * num_steps;
    std::cout << ops / 1e9 << std::endl;

    t_end = omp_get_wtime();
    auto divisor = (t_end - t_start) * 1e9;
    std::cout << "Time: " << t_end - t_start << "\n";
    std::cout << "OPS: " << ops << std::endl;
    std::cout << "GFLOPS: " << ops / divisor << "\n";
}
