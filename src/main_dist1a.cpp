#include "spmv.hpp"
#include <csr.hpp>
#include <iostream>
#include <metis.h>
#include <mpi.h>
#include <mtx.hpp>
int main(int argc, char **argv) {
    int rank, size;
    int num_steps = 100;
    double t_start, t_end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSR<int, double> csr;
    std::vector<double> A, y;
    std::vector<int> p(size + 1, 0);

    if (rank == 0) {
        std::string file = argv[1];
        MTX<int, double> mtx;
        mtx.read_mtx(file);
        csr = mtx.mtx_to_csr();
        A.assign(csr.V, 0);

        for (int i = 0; i < csr.V; i++)
            A[i] = 1;
        csr.partition(size, p, A);
    }
    unsigned long long int ops = csr.nnz * 2 * num_steps;

    MPI_Barrier(MPI_COMM_WORLD);
    csr.distribute(rank, size, p, A);
    y.assign(csr.V, 0);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(p.data(), size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> recvcounts(size, 0);
    std::adjacent_difference(p.begin() + 1, p.end(), recvcounts.begin());

    t_start = MPI_Wtime();
    while (num_steps--) {
        spmv(csr, A, y);
        MPI_Allgatherv(y.data(), csr.V, MPI_DOUBLE, A.data(), recvcounts.data(), p.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    t_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Time: " << t_end - t_start << "\n";
        std::cout << "GFLOPS: " << ops / ((t_end - t_start) * 1e9) << "\n";
    }

    MPI_Finalize();

    return 0;
}
