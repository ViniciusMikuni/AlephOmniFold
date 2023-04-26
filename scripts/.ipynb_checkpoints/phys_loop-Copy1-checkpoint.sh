#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4

export SLURM_CPU_BIND="cores"
for i in {00..40};do srun -n 1 python /pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/scripts/aleph.py --run_id=$i --strapn=10; done
