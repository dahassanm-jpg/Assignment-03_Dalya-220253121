/* Assignment 03 - Dalya Awad Mohammad Hassan 20253121 - Task 3: Distributed matrix multiplication C = A x B, N = 2048,
 * row-block decomposition.
 *   MPI_Bcast   : B from rank 0 to all
 *   MPI_Scatter : rows of A
 *   local compute of the rows of C
 *   MPI_Gather  : rows of C back to rank 0
 * Rank 0 verifies against a serial reference (not included in timings).
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEFAULT_N 2048

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "N=%d must be divisible by np=%d\n", N, size);
        MPI_Finalize();
        return 1;
    }
    int rows_per_rank = N / size;

    double *A = NULL, *C = NULL;
    double *B       = malloc((size_t)N * N * sizeof(double));
    double *local_A = malloc((size_t)rows_per_rank * N * sizeof(double));
    double *local_C = calloc((size_t)rows_per_rank * N, sizeof(double));

    if (rank == 0) {
        A = malloc((size_t)N * N * sizeof(double));
        C = malloc((size_t)N * N * sizeof(double));
        srand(12345);
        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_rank * N, MPI_DOUBLE,
                local_A, rows_per_rank * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    double t_comm = MPI_Wtime() - t0;

    /* local compute (i-k-j order for cache-friendly access to B) */
    t0 = MPI_Wtime();
    for (int i = 0; i < rows_per_rank; i++) {
        for (int k = 0; k < N; k++) {
            double a = local_A[(size_t)i * N + k];
            const double *brow = &B[(size_t)k * N];
            double *crow = &local_C[(size_t)i * N];
            for (int j = 0; j < N; j++)
                crow[j] += a * brow[j];
        }
    }
    double t_comp = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_Gather(local_C, rows_per_rank * N, MPI_DOUBLE,
               C, rows_per_rank * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_start;

    /* slowest rank defines each phase */
    double max_total, max_comp, max_comm;
    MPI_Reduce(&t_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp,  &max_comp,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,  &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* verification against serial reference on rank 0 (untimed) */
    int ok = 1;
    if (rank == 0) {
        double *R = calloc((size_t)N * N, sizeof(double));
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double a = A[(size_t)i * N + k];
                for (int j = 0; j < N; j++)
                    R[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        double maxerr = 0.0;
        for (size_t i = 0; i < (size_t)N * N; i++) {
            double e = fabs(R[i] - C[i]);
            if (e > maxerr) maxerr = e;
        }
        ok = (maxerr < 1e-9);
        fprintf(stderr, "np=%d max abs error vs serial = %.3e -> %s\n",
                size, maxerr, ok ? "PASS" : "FAIL");
        printf("%d,%d,%.4f,%.4f,%.4f,%s\n", size, N, max_total, max_comp, max_comm,
               ok ? "PASS" : "FAIL");
        free(R); free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return ok ? 0 : 2;
}
