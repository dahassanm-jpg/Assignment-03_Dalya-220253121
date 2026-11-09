#!/bin/bash
#SBATCH --job-name=deadlock
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=00:03:00
#SBATCH --output=deadlock_%j.out
#SBATCH --error=deadlock_%j.err

# module load openmpi
mpicc -O2 -o deadlock deadlock.c

echo "=== 1. Original code (expected to HANG; killed after 10 s) ==="
timeout 10 mpirun -np 2 ./deadlock deadlock
echo "exit status: $? (124 = killed by timeout, i.e. deadlock)"

for mode in reorder sendrecv nonblocking; do
    echo
    echo "=== Fix: $mode ==="
    timeout 30 mpirun -np 2 ./deadlock $mode
    echo "exit status: $?"
done
