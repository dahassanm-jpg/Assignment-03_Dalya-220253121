#!/bin/bash
#SBATCH --job-name=hello_mpi
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --output=hello_mpi_%j.out
#SBATCH --error=hello_mpi_%j.err


mpicc -O2 -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi
# (alternatively: srun -n 4 ./hello_mpi)
