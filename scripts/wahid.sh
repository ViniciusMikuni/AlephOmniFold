#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4

export CUDA_VISIBLE_DEVICES=0
for i in {1..5}
do
  srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph.py --run_id=$i &
done

export CUDA_VISIBLE_DEVICES=1
for i in {1..5}
do
  srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph.py --run_id=$i &
done

config="poisadv-asym-rs"
id="khwl7w32"
export CUDA_VISIBLE_DEVICES=2
for i in {1..5}
do
  srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph.py --run_id=$i &
done

export CUDA_VISIBLE_DEVICES=3
for i in {1..5}
do
  srun -n 1 python /global/homes/m/mavaylon/phys/OmniFold/scripts/aleph.py --run_id=$i &
done