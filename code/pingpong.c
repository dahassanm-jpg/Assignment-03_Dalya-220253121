/* Assignment 03 - Dalya Awad Mohammad Hassan 20253121 - Task 2: Ping-pong latency/bandwidth between ranks 0 and 1.
 * Part A: blocking MPI_Send / MPI_Recv
 * Part B: nonblocking MPI_Isend / MPI_Irecv + MPI_Wait
 * Sizes: 1 KB, 64 KB, 1 MB, 4 MB; 100 timed iterations (+10 warm-up).
 * Bandwidth = size / (avg round-trip / 2).
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS  100
#define WARMUP 10

static void pingpong_blocking(char *buf, int n, int rank)
{
    if (rank == 0) {
        MPI_Send(buf, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, n, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(buf, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, n, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
}

static void pingpong_nonblocking(char *buf, int n, int rank)
{
    MPI_Request req;
    if (rank == 0) {
        MPI_Isend(buf, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Irecv(buf, n, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {
        MPI_Irecv(buf, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Isend(buf, n, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Need at least 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    const int sizes[] = {1024, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    const int nsizes = 4;
    char *buf = malloc(4 * 1024 * 1024);
    memset(buf, 1, 4 * 1024 * 1024);

    if (rank == 0)
        printf("mode,size_bytes,avg_rtt_us,one_way_us,bandwidth_MBps\n");

    for (int mode = 0; mode < 2; mode++) {
        for (int s = 0; s < nsizes; s++) {
            int n = sizes[s];
            if (rank > 1) continue;   /* only ranks 0 and 1 participate */

            for (int i = 0; i < WARMUP; i++)
                mode ? pingpong_nonblocking(buf, n, rank) : pingpong_blocking(buf, n, rank);

            MPI_Barrier(MPI_COMM_WORLD);   /* ranks 0,1 only when size==2 */
            double t0 = MPI_Wtime();
            for (int i = 0; i < ITERS; i++)
                mode ? pingpong_nonblocking(buf, n, rank) : pingpong_blocking(buf, n, rank);
            double t1 = MPI_Wtime();

            if (rank == 0) {
                double rtt = (t1 - t0) / ITERS;               /* seconds */
                double bw  = n / (rtt / 2.0) / 1e6;           /* MB/s    */
                printf("%s,%d,%.3f,%.3f,%.2f\n",
                       mode ? "nonblocking" : "blocking",
                       n, rtt * 1e6, rtt * 0.5e6, bw);
            }
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
