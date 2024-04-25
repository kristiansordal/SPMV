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

    int num_steps = 100;
    double start, end;
    unsigned long long ops;

    for (int i = 0; i < g.N; i++)
        A[i] = ((double)rand() / (RAND_MAX)) + 1;

    start = omp_get_wtime();
    for (int i = 0; i < num_steps; i++) {
        spmv_shared(g, A, y);
        std::swap(A, y);
    }

#pragma omp parallel
    { std::cout << "Hello from thread: " << omp_get_thread_num() << std::endl; }

    ops = 2 * g.nnz * num_steps;
    end = omp_get_wtime();
    std::cout << "Time: " << end - start << "\nGFLOPS: " << ops / ((end - start) * 1e9) << "\n";
}
