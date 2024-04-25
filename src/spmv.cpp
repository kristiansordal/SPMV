#include "spmv.hpp"
#include <omp.h>
void spmv(Graph<int, double> &g, std::vector<double> &A, std::vector<double> &y) {
    for (int v = 0; v < g.V; v++) {
        double sum = 0;
        for (int u = g.row_ptr[v]; u < g.row_ptr[v + 1]; u++)
            sum += g.vals[u] * A[g.col_idx[u]];
        y[v] = sum;
    }
}

void spmv_shared(Graph<int, double> &g, std::vector<double> &A, std::vector<double> &y) {
#pragma omp parallel for
    for (int v = 0; v < g.V - 1; v++) {
        double sum = 0;
        for (int u = g.row_ptr[v]; u < g.row_ptr[v + 1]; u++)
            sum += g.vals[u] * A[g.col_idx[u]];
        y[v] = sum;
    }
}
