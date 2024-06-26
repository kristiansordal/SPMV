#include <csr.hpp>
#include <mpi.h>
#include <mtx.hpp>
#define STEPS 100
using namespace std;
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    CSR<int, double> csr;
    string file = argv[1];
    vector<int> p(size + 1, 0);
    vector<double> v_old;
    vector<vector<int>> separators(size);
    if (rank == 0) {
        MTX<int, double> mtx;
        mtx.read_mtx(file);
        csr = mtx.mtx_to_csr();

        v_old.assign(csr.N, 0);
        for (int i = 0; i < csr.N; i++)
            v_old[i] = i;

        csr.partition(size, p, v_old);
        csr.determine_separators(size, p, separators);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    csr.distribute(rank, size, p, v_old);
    MPI_Finalize();
    return 0;
}
