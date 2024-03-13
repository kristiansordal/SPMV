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

    char *data = mmap(0, size, PROT_READ, MAP_SHARED, fileno_unlocked(f), 0);
    size_t p = 0;

    mtx m = internal_parse_mtx_header(data, &p);

    while (data[p] == '%')
        skip_line_safe(data, &p, size);

    parse_int(data, &p, &m.M);
    parse_int(data, &p, &m.N);
    parse_int(data, &p, &m.L);

    m.I = malloc(sizeof(int) * m.L);
    m.J = malloc(sizeof(int) * m.L);
    m.A = malloc(sizeof(double) * m.L);

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
            tc = malloc(sizeof(int) * nt);

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

    m.I = malloc(sizeof(int) * m.L);
    m.J = malloc(sizeof(int) * m.L);
    m.A = malloc(sizeof(double) * m.L);

    for (int i = 0; i < m.L; i++) {
        rc = getline(&line, &size, f);
        p = 0;

        parse_int(line, &p, m.I + i);
        parse_int(line, &p, m.J + i);
        parse_real(line, &p, m.A + i);
    }

    free(line);

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

Graph parse_mtx(FILE *f) {
    mtx m = internal_parse_mtx_seq(f);

    Graph G;
    G.n = m.N > m.M ? m.N : m.M;
    G.vertices = calloc(G.n + 1, sizeof(int));

    // Count degree

#pragma omp parallel for
    for (int i = 0; i < m.L; i++) {
        __atomic_add_fetch(G.vertices + (m.I[i] - 1), 1, __ATOMIC_RELAXED);

        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
            __atomic_add_fetch(G.vertices + (m.J[i] - 1), 1, __ATOMIC_RELAXED);

        // G.vertices[m.I[i] - 1]++;
        // if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
        //     G.vertices[m.J[i] - 1]++;
    }

    for (int i = 1; i <= G.n; i++) {
        G.vertices[i] += G.vertices[i - 1];
    }

    G.m = G.vertices[G.n];
    G.edges = malloc(sizeof(int) * G.m);
    G.vals = malloc(sizeof(double) * G.m);

#pragma omp parallel for
    for (int i = 0; i < m.L; i++) {
        int j = __atomic_sub_fetch(G.vertices + (m.I[i] - 1), 1, __ATOMIC_RELAXED);
        G.edges[j] = m.J[i] - 1;
        G.vals[j] = m.A[i];

        if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC) {
            j = __atomic_sub_fetch(G.vertices + (m.J[i] - 1), 1, __ATOMIC_RELAXED);
            G.edges[j] = m.I[i] - 1;
            G.vals[j] = m.A[i];
        }

        // G.vertices[m.I[i] - 1]--;
        // G.edges[G.vertices[m.I[i] - 1]] = m.J[i] - 1;
        // G.vals[G.vertices[m.I[i] - 1]] = m.A[i];

        // if (m.I[i] != m.J[i] && m.symmetry == SYMMETRIC)
        // {
        //     G.vertices[m.J[i] - 1]--;
        //     G.edges[G.vertices[m.J[i] - 1]] = m.I[i] - 1;
        //     G.vals[G.vertices[m.J[i] - 1]] = m.A[i];
        // }
    }

    internal_free_mtx(&m);

    return G;
}

Graph parse_and_validate_mtx(const char *path) {
    FILE *f = fopen(path, "r");
    Graph G = parse_mtx(f);
    fclose(f);

    printf("|V|=%d |E|=%d\n", G.n, G.m);

    normalize_graph(G);
    sort_edges(G);
    if (!validate_graph(G))
        printf("Error in graph\n");

    return G;
}

void free_graph(Graph *G) {
    G->n = 0;
    G->m = 0;
    free(G->vals);
    free(G->vertices);
    free(G->edges);
    G->vals = NULL;
    G->vertices = NULL;
    G->edges = NULL;
}

int compare(const void *a, const void *b, void *c) {
    int ia = *(const int *)a, ib = *(const int *)b;
    int *data = (int *)c; // Cast void* back to int* to use it as intended.
    printf("%d %d\n", ia, ib);

    return data[ia] - data[ib];
}

void sort_edges(Graph G) {

#pragma omp parallel
    {
        int *index = malloc(sizeof(int) * G.m);
        int *E_buffer = malloc(sizeof(int) * G.m);
        double *A_buffer = malloc(sizeof(double) * G.m);

#pragma omp for
        for (int u = 0; u < G.n; u++) {
            int degree = G.vertices[u + 1] - G.vertices[u];
            for (int i = 0; i < degree; i++) {
                index[i] = i;
            }

#ifdef __APPLE__
            qsort_r(index, degree, sizeof(int), G.edges + G.vertices[u], compare);
#elif __linux__
            qsort_r(index, degree, sizeof(int), compare, G.edges + G.vertices[u]);
#endif

            for (int i = 0; i < degree; i++) {
                E_buffer[i] = G.edges[G.vertices[u] + index[i]];
                A_buffer[i] = G.vals[G.vertices[u] + index[i]];
            }

            memcpy(G.edges + G.vertices[u], E_buffer, degree * sizeof(int));
            memcpy(G.vals + G.vertices[u], A_buffer, degree * sizeof(double));
        }

        free(index);
        free(E_buffer);
        free(A_buffer);
    }
    printf("Graph sorted\n");
}

void normalize_graph(Graph G) {
    double mean = 0.0;
#pragma omp parallel for reduction(+ : mean)
    for (int i = 0; i < G.m; i++) {
        mean += G.vals[i];
    }

    if (mean == 0.0) // All zero input
    {
#pragma omp parallel for
        for (int i = 0; i < G.m; i++)
            G.vals[i] = 2.0;
        return;
    }

    mean /= (double)G.m;

    double std = 0.0;
#pragma omp parallel for reduction(+ : std)
    for (int i = 0; i < G.m; i++) {
        std += (G.vals[i] - mean) * (G.vals[i] - mean);
    }

    std = sqrt(std / (double)G.m);

#pragma omp parallel for
    for (int i = 0; i < G.m; i++) {
        G.vals[i] = (G.vals[i] - mean) / (std + __DBL_EPSILON__);
    }
    printf("Graph normalized\n");
}

int validate_graph(Graph G) {
    for (int u = 0; u < G.n; u++) {
        int degree = G.vertices[u + 1] - G.vertices[u];
        if (degree < 0 || degree > G.m)
            return 0;

        for (int i = G.vertices[u]; i < G.vertices[u + 1]; i++) {
            if (G.edges[i] < 0 || G.edges[i] > G.m)
                return 0;
            if (i > G.vertices[u] && G.edges[i] <= G.edges[i - 1])
                return 0;
        }
    }
    return 1;
}
