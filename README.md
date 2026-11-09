# Dalya Awad Mohammad Hassan 20253121 - Assignment 03: MPI Programming

All programs were developed and run on Ubuntu (WSL2) with OpenMPI and a
single-node Slurm (`slurmctld` + `slurmd`), 24 logical CPUs.

## Files

| Task | Source | Slurm script |
|---|---|---|
| 1 Hello World | `hello_mpi.c` | `hello_mpi.slurm` |
| 2 Ping-pong | `pingpong.c` | `pingpong.slurm` |
| 3 Matrix multiply | `matmul_mpi.c` | `matmul_mpi.slurm` (np = 1,2,4,8,16), `matmul_repeat.slurm` (3 repetitions) |
| 4 Deadlock | `deadlock.c` | `deadlock.slurm` |
| 5 Scaling | `analyze_scaling.py` | uses output of `matmul_repeat.slurm` |

Outputs of the submitted jobs are kept as `*_<jobid>.out` / `.err`.
The report is `Assignment3_Report.docx`.

## Build and run (interactive)

```bash
mpicc -O2 -o hello_mpi hello_mpi.c       && mpirun -np 4 ./hello_mpi
mpicc -O2 -o pingpong pingpong.c         && mpirun -np 2 ./pingpong
mpicc -O3 -march=native -o matmul_mpi matmul_mpi.c -lm
mpirun -np 4 ./matmul_mpi 2048           # prints: np,N,total_s,compute_s,comm_s,verified
mpicc -O2 -o deadlock deadlock.c
timeout 10 mpirun -np 2 ./deadlock deadlock      # hangs -> killed by timeout (exit 124)
mpirun -np 2 ./deadlock reorder                  # also: sendrecv, nonblocking
```

## Submit with Slurm

```bash
sbatch hello_mpi.slurm
sbatch pingpong.slurm
sbatch matmul_mpi.slurm
sbatch deadlock.slurm
sbatch matmul_repeat.slurm
squeue                      # watch the queue
```
On a cluster, uncomment/adjust the `module load openmpi` line in each script.

## Scaling analysis

```bash
python analyze_scaling.py matmul_repeat_<jobid>.out
```
Needs Python with matplotlib. Produces `speedup.png`, `efficiency.png`,
`overhead.png` and `scaling_table.md` (medians of 3 runs).

## Notes
- `matmul_mpi` requires N divisible by the number of processes.
- The matmul program verifies against a serial reference on rank 0 (not timed).
- `deadlock.c` uses `MPI_Ssend` and 8 MB messages so the deadlock is deterministic.
