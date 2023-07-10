#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4

export SLURM_CPU_BIND="cores"
srun -n 1 python /pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/scripts/nested_directory_maker.py --run_id=$RUN_ID --strapn=$STRAPN;
