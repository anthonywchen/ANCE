#!/bin/bash
#
#SBATCH --job-name=ann_data_gen
#SBATCH --time=48:00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=20
#SBATCH --gpus=4
#SBATCH --mem=150GB

cd commands/
srun ./run_ann_data_gen.sh
