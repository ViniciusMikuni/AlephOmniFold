#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4

export SLURM_CPU_BIND="cores"
for i in {00..100};do srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph_pzh.py --run_id=$i ; done
