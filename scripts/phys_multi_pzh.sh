#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1


module load python
conda activate phys1
module load tensorflow/2.6.0

export SLURM_CPU_BIND="cores"
srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph_pzh.py --run_id=0 &> run0.out &
srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph_pzh.py --run_id=1 &> run1.out &
srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph_pzh.py --run_id=2 &> run2.out &
srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph_pzh.py --run_id=3 &> run3.out &
wait
