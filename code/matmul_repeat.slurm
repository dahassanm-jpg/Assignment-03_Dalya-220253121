#!/bin/bash
#SBATCH --job-name=matmul_rep
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=matmul_repeat_%j.out
#SBATCH --error=matmul_repeat_%j.err

# Task 5 data: 3 repetitions per process count (medians are used in the analysis)
# module load openmpi
mpicc -O3 -march=native -o matmul_mpi matmul_mpi.c -lm

echo "np,N,total_s,compute_s,comm_s,verified"
for rep in 1 2 3; do
    for np in 1 2 4 8 16; do
        mpirun -np $np ./matmul_mpi 2048
    done
done
