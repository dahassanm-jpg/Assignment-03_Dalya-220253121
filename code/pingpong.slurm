#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=pingpong_%j.out
#SBATCH --error=pingpong_%j.err

# module load openmpi
mpicc -O2 -o pingpong pingpong.c
mpirun -np 2 ./pingpong
