#pragma once
#include <graph.hpp>
#include <vector>
void spmv(Graph<int, double> &g, std::vector<double> &A, std::vector<double> &y);
void spmv_shared(Graph<int, double> &g, std::vector<double> &A, std::vector<double> &y);
