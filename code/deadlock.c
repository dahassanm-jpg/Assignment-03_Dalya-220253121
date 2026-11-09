/* Assignment 03 - Dalya Awad Mohammad Hassan 20253121 - Task 4: Deadlock demonstration and three fixes.
 *
 *   ./deadlock deadlock    both ranks send first -> hangs
 *   ./deadlock reorder     rank 0 sends then receives; rank 1 receives then sends
 *   ./deadlock sendrecv    MPI_Sendrecv
 *   ./deadlock nonblocking MPI_Isend / MPI_Irecv + MPI_Waitall
 *
 * Why it deadlocks: MPI_Send may return only after the receiver has started
 * matching the message (rendezvous protocol for large messages). Both ranks
 * sit inside MPI_Send waiting for a receive that neither has posted yet.
 * To make this deterministic (small messages are buffered "eagerly" and
 * would not hang), the demonstration uses a large message (COUNT doubles =
 * 8 MB) and the synchronous MPI_Ssend, which never completes before the
 * matching receive is posted.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT (1 << 20)   /* 1M doubles = 8 MB */

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 ranks\n");
        MPI_Finalize();
        return 1;
    }
    const char *mode = (argc > 1) ? argv[1] : "deadlock";
    int other = 1 - rank;

    double *sendbuf = malloc(COUNT * sizeof(double));
    double *recvbuf = malloc(COUNT * sizeof(double));
    for (int i = 0; i < COUNT; i++) { sendbuf[i] = rank; recvbuf[i] = -1; }

    printf("[rank %d] mode=%s: starting\n", rank, mode);
    fflush(stdout);

    if (strcmp(mode, "deadlock") == 0) {
        /* BOTH ranks send first -> circular wait */
        MPI_Ssend(sendbuf, COUNT, MPI_DOUBLE, other, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, COUNT, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (strcmp(mode, "reorder") == 0) {
        /* Fix 1: break the symmetry */
        if (rank == 0) {
            MPI_Ssend(sendbuf, COUNT, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, COUNT, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, COUNT, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Ssend(sendbuf, COUNT, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    } else if (strcmp(mode, "sendrecv") == 0) {
        /* Fix 2: MPI handles the pairing internally */
        MPI_Sendrecv(sendbuf, COUNT, MPI_DOUBLE, other, 0,
                     recvbuf, COUNT, MPI_DOUBLE, other, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (strcmp(mode, "nonblocking") == 0) {
        /* Fix 3: post both operations, then wait */
        MPI_Request reqs[2];
        MPI_Irecv(recvbuf, COUNT, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sendbuf, COUNT, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    } else {
        if (rank == 0) fprintf(stderr, "unknown mode '%s'\n", mode);
        MPI_Finalize();
        return 1;
    }

    printf("[rank %d] mode=%s: completed, received value %.0f from rank %d\n",
           rank, mode, recvbuf[0], other);

    free(sendbuf); free(recvbuf);
    MPI_Finalize();
    return 0;
}
