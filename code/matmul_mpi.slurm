#!/bin/bash
#SBATCH --job-name=matmul_mpi
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=matmul_mpi_%j.out
#SBATCH --error=matmul_mpi_%j.err

# module load openmpi
mpicc -O3 -march=native -o matmul_mpi matmul_mpi.c -lm

echo "np,N,total_s,compute_s,comm_s,verified"
# 16 is needed for Task 5
for np in 1 2 4 8 16; do
    mpirun -np $np ./matmul_mpi 2048
done
