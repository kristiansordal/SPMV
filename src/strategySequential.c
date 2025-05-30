#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
int main(int argc, char **argv) {
    CSR g;
    clock_t start, end;
    g = parse_and_validate_mtx(argv[1]);

    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 2.0;
        y[i] = 2.0;
    }
    long long int flops = 0;

    start = clock();
    for (int i = 0; i < 100; i++) {
        spmv(g, x, y, &flops);
        double *tmp = x;
        x = y;
        y = tmp;
    }
    end = clock();
    double ops = (long long)g.num_cols * 2ll * 100ll;

    double l2 = 0.0;
    for (int i = 0; i < g.num_rows; i++)
        l2 += x[i] * x[i];
    l2 = sqrt(l2);

    double time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("L2 norm: %f\n", l2);
    printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("GFLOPS: %f\n", (ops / (time * 1e9)));

    free(x);
    free(y);
    return 0;
}
