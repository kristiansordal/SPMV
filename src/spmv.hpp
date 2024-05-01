#pragma once
#include <csr.hpp>
#include <vector>
void spmv(CSR<int, double> &g, std::vector<double> &A, std::vector<double> &y);
void spmv_shared(CSR<int, double> &g, std::vector<double> &A, std::vector<double> &y);
double l2_norm(CSR<int, double> &g);
