#include "spmv.hpp"
#include <omp.h>
void spmv(CSR<int, double> &csr, std::vector<double> &A, std::vector<double> &y) {
    for (int v = 0; v < csr.V; v++) {
        double sum = 0;
        for (int u = csr.row_ptr[v]; u < csr.row_ptr[v + 1]; u++)
            sum += csr.vals[u] * A[csr.col_idx[u]];

        y[v] = sum;
    }
}

void spmv_shared(CSR<int, double> &csr, std::vector<double> &A, std::vector<double> &y) {
#pragma omp parallel for schedule(runtime)
    for (int v = 0; v < csr.V; v++) {
        double sum = 0;
        for (int u = csr.row_ptr[v]; u < csr.row_ptr[v + 1]; u++)
            sum += csr.vals[u] * A[csr.col_idx[u]];
        y[v] = sum;
    }
}

double l2_norm(CSR<int, double> &g) {
    double norm = 0;
    for (int i = 0; i < g.V; i++)
        norm += g.vals[i] * g.vals[i];
    return sqrt(norm);
}
