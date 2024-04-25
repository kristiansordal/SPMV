#include <graph.hpp>
#include <mpi.h>
#include <mtx.hpp>
#define STEPS 100
using namespace std;
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Graph<int, double> g;
    string file = argv[1];
    vector<int> p(size + 1, 0);
    vector<double> v_old;
    vector<vector<int>> separators(size);
    if (rank == 0) {
        MTX<int, double> mtx;
        mtx.read_mtx(file);
        g = mtx.mtx_to_csr();

        v_old.assign(g.N, 0);
        for (int i = 0; i < g.N; i++)
            v_old[i] = i;

        g.partition_graph(size, p, v_old);
        g.determine_separators(size, p, separators);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    g.distribute_graph(rank, size, p);
    MPI_Finalize();
    return 0;
}
