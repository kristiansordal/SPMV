#include "mtx.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define GENERAL 0
#define SYMMETRIC 1

typedef struct {
    int symmetry;
    int M, N, L;
    int *I, *J;
    double *A;
} mtx;

static inline void parse_int(char *data, size_t *p, int *v) {
    while (data[*p] == ' ')
        (*p)++;

    *v = 0;

    int sign = 1;
    if (data[*p] == '-') {
        sign = -1;
        (*p)++;
    }

    while (data[*p] >= '0' && data[*p] <= '9') {
        *v = (*v) * 10 + data[*p] - '0';
        (*p)++;
    }

    *v *= sign;
}

static inline void parse_real(char *data, size_t *p, double *v) {
    while (data[*p] == ' ')
        (*p)++;

    *v = 0.0;

    double sign = 1.0;
    if (data[*p] == '-') {
        sign = -1.0;
        (*p)++;
    }

    while (data[*p] >= '0' && data[*p] <= '9') {
        *v = (*v) * 10.0 + (double)(data[*p] - '0');
        (*p)++;
    }

    if (data[*p] == '.') {
        (*p)++;
        double s = 0.1;
        while (data[*p] >= '0' && data[*p] <= '9') {
            *v += (double)(data[*p] - '0') * s;
            (*p)++;
            s *= 0.1;
        }
    }
    *v *= sign;

    if (data[*p] == 'e') {
        (*p)++;
        int m;
        parse_int(data, p, &m);
        *v *= pow(10.0, m);
    }
}

static inline void skip_line(char *data, size_t *p) {
    while (data[*p] != '\n')
        (*p)++;
    (*p)++;
}

static inline void skip_line_safe(char *data, size_t *p, size_t t) {
    while (*p < t && data[*p] != '\n')
        (*p)++;
    (*p)++;
}

mtx internal_parse_mtx_header(char *data, size_t *p) {
    mtx m;

    char header[256];
    while (*p < 255 && data[*p] != '\n') {
        header[*p] = data[*p];
        (*p)++;
    }
    header[*p] = '\0';
    (*p)++;

    if (*p == 256) {
        fprintf(stderr, "Invalid header %s\n", header);
        exit(1);
    }

    char *token = strtok(header, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");
    token = strtok(NULL, " ");

    if (strcmp(token, "general") == 0)
        m.symmetry = GENERAL;
    else if (strcmp(token, "symmetric") == 0)
        m.symmetry = SYMMETRIC;
    else {
        fprintf(stderr, "Invalid symmetry %s\n", token);
        exit(1);
    }

    return m;
}

mtx internal_parse_mtx(FILE *f) {
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = (char *)mmap(0, size, PROT_READ, MAP_SHARED, fileno_unlocked(f), 0);
    size_t p = 0;

    mtx m = internal_parse_mtx_header(data, &p);

    while (data[p] == '%')
        skip_line_safe(data, &p, size);

    parse_int(data, &p, &m.M);
    parse_int(data, &p, &m.N);
    parse_int(data, &p, &m.L);

    m.I = (int *)malloc(sizeof(int) * m.L);
    m.J = (int *)malloc(sizeof(int) * m.L);
    m.A = (double *)malloc(sizeof(double) * m.L);

    int *tc;

#pragma omp parallel shared(tc) firstprivate(p, size, data, m)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        size_t s = ((size - p) / nt) * tid + p;
        size_t t = s + (size - p) / nt;
        if (tid == nt - 1)
            t = size;

        if (tid == 0)
            tc = (int *)malloc(sizeof(int) * nt);

#pragma omp barrier

        int lc = 0;
        for (size_t i = s; i < t; i++)
            if (data[i] == '\n')
                lc++;
        tc[tid] = lc;

#pragma omp barrier

        p = s;
        s = 0;
        for (int i = 0; i < tid; i++)
            s += tc[i];

        t = s + tc[tid];
        if (tid == nt - 1 || t > m.L)
            t = m.L;

        for (int i = s; i < t; i++) {
            skip_line_safe(data, &p, size);

            parse_int(data, &p, m.I + i);
            parse_int(data, &p, m.J + i);
            parse_real(data, &p, m.A + i);
        }

#pragma omp barrier

        if (tid == 0)
            free(tc);
    }

    // for (int i = 0; i < m.L; i++)
    // {
    //     skip_line(data, &p);

    //     while (data[p] == '%')
    //         skip_line(data, &p);

    //     parse_int(data, &p, m.I + i);
    //     parse_int(data, &p, m.J + i);
    //     parse_real(data, &p, m.A + i);
    // }

    munmap(data, size);

    return m;
}

mtx internal_parse_mtx_seq(FILE *f) {
    char *line = NULL;
    size_t size = 0, rc = 0, p = 0;
    rc = getline(&line, &size, f);
    mtx m = internal_parse_mtx_header(line, &p);
    rc = getline(&line, &size, f);

    while (line[0] == '%')
        rc = getline(&line, &size, f);

    p = 0;
    parse_int(line, &p, &m.M);
    parse_int(line, &p, &m.N);
    parse_int(line, &p, &m.L);

    m.I = (int *)malloc(sizeof(int) * m.L);
    m.J = (int *)malloc(sizeof(int) * m.L);
    m.A = (double *)malloc(sizeof(double) * m.L);

    for (int i = 0; i < m.L; i++) {
        rc = getline(&line, &size, f);
        p = 0;

        parse_int(line, &p, m.I + i);
        parse_int(line, &p, m.J + i);
        parse_real(line, &p, m.A + i);
    }

    free(line);
    printf("Done parsing mtx\n");
    fflush(stdout);

    return m;
}

void internal_free_mtx(mtx *m) {
    m->symmetry = GENERAL;
    m->M = 0;
    m->N = 0;
    m->L = 0;
    free(m->I);
    free(m->J);
    free(m->A);
    m->I = NULL;
    m->J = NULL;
    m->A = NULL;
}

CSR parse_mtx(FILE *f) {
    mtx m = internal_parse_mtx_seq(f);
    printf("%d, %d, %d\n", m.M, m.N, m.L);

    CSR g;
    g.num_rows = m.N > m.M ? m.N : m.M;
    g.row_ptr = (int *)calloc(g.num_rows + 1, sizeof(int));

    // Count degree

#pragma omp parallel for
    for (int i = 0; i < m.L; i++) {
        __atomic_add_fetch(g.row_ptr + (m.I[i] - 1), 1, __ATOMIC_RELAXED);

        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
            __atomic_add_fetch(g.row_ptr + (m.J[i] - 1), 1, __ATOMIC_RELAXED);

        // g.V[m.I[i] - 1]++;
        // if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
        //     g.V[m.J[i] - 1]++;
    }

    for (int i = 1; i <= g.num_rows; i++) {
        g.row_ptr[i] += g.row_ptr[i - 1];
    }

    g.num_cols = g.row_ptr[g.num_rows];
    g.col_idx = (int *)malloc(sizeof(int) * g.num_cols);
    g.values = (double *)malloc(sizeof(double) * g.num_cols);

#pragma omp parallel for
    for (int i = 0; i < m.L; i++) {
        int j = __atomic_sub_fetch(g.row_ptr + (m.I[i] - 1), 1, __ATOMIC_RELAXED);
        g.col_idx[j] = m.J[i] - 1;
        g.values[j] = m.A[i];

        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC) {
            j = __atomic_sub_fetch(g.row_ptr + (m.J[i] - 1), 1, __ATOMIC_RELAXED);
            g.col_idx[j] = m.I[i] - 1;
            g.values[j] = m.A[i];
        }

        // g.V[m.I[i] - 1]--;
        // g.E[g.V[m.I[i] - 1]] = m.J[i] - 1;
        // g.A[g.V[m.I[i] - 1]] = m.A[i];

        // if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
        // {
        //     g.V[m.J[i] - 1]--;
        //     g.E[g.V[m.J[i] - 1]] = m.I[i] - 1;
        //     g.A[g.V[m.J[i] - 1]] = m.A[i];
        // }
    }

    internal_free_mtx(&m);

    return g;
}

CSR parse_and_validate_mtx(const char *path) {
    FILE *f = fopen(path, "r");
    CSR g = parse_mtx(f);
    fclose(f);

    printf("|V|=%d |E|=%d\n", g.num_rows, g.num_cols);

    printf("Normalizing graph\n");
    normalize_graph(g);
    printf("Sorting edges\n");
    sort_edges(g);
    if (!validate_graph(g))
        printf("Error in graph\n");

    return g;
}

void free_graph(CSR *g) {
    g->num_rows = 0;
    g->num_cols = 0;
    free(g->values);
    free(g->row_ptr);
    free(g->col_idx);
    g->values = NULL;
    g->row_ptr = NULL;
    g->col_idx = NULL;
}

int compare(const void *a, const void *b, void *c) {
    int ia = *(int *)a, ib = *(int *)b;
    int *data = (int *)c;

    return data[ia] - data[ib];
}

void sort_edges(CSR g) {
#pragma omp parallel
    {
        int *index = (int *)malloc(sizeof(int) * g.num_rows);
        int *E_buffer = (int *)malloc(sizeof(int) * g.num_rows);
        double *A_buffer = (double *)malloc(sizeof(double) * g.num_rows);

#pragma omp for
        for (int u = 0; u < g.num_rows; u++) {
            int degree = g.row_ptr[u + 1] - g.row_ptr[u];

            for (int i = 0; i < degree; i++)
                index[i] = i;

            qsort_r(index, degree, sizeof(int), compare, g.col_idx + g.row_ptr[u]);
            for (int i = 0; i < degree; i++) {
                E_buffer[i] = g.col_idx[g.row_ptr[u] + index[i]];
                A_buffer[i] = g.values[g.row_ptr[u] + index[i]];
            }

            memcpy(g.col_idx + g.row_ptr[u], E_buffer, degree * sizeof(int));
            memcpy(g.values + g.row_ptr[u], A_buffer, degree * sizeof(double));
        }

        free(index);
        free(E_buffer);
        free(A_buffer);
    }
}

int cmpfunc(const void *a, const void *b) { return (*(double *)a - *(double *)b); }

void normalize_graph(CSR g) {
    double mean = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : mean)
    for (int i = 0; i < g.num_cols; i++) {
        mean += g.values[i];
    }

    printf("Mean of graph: %f\n", mean);
    fflush(stdout);

    if (mean == 0.0) // All zero input
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < g.num_cols; i++)
            g.values[i] = 2.0;
        return;
    }

    mean /= (double)g.num_cols;

    double std = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : std)
    for (int i = 0; i < g.num_cols; i++) {
        std += (g.values[i] - mean) * (g.values[i] - mean);
    }

    std = sqrt(std / (double)g.num_cols);
    printf("Std of graph: %f\n", std);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < g.num_cols; i++) {
        g.values[i] = (g.values[i] - mean) / (std + __DBL_EPSILON__);
    }
}

int validate_graph(CSR g) {
    for (int u = 0; u < g.num_rows; u++) {
        int degree = g.row_ptr[u + 1] - g.row_ptr[u];
        if (degree < 0 || degree > g.num_cols) {
            printf("Invalid degree: %d\n", degree);
            return 0;
        }

        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            if (g.col_idx[i] < 0 || g.col_idx[i] > g.num_cols) {
                printf("Invalid column index: %d\n", g.col_idx[i]);
                return 0;
            }
            if (i > g.row_ptr[u] && g.col_idx[i] <= g.col_idx[i - 1]) {
                printf("Invalid column index order: %d %d\n", g.col_idx[i], g.col_idx[i - 1]);
                return 0;
            }
        }
    }
    return 1;
}
