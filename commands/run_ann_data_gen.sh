#!/bin/bash
#
#SBATCH --job-name=train_ance
#SBATCH --time=48:00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=25
#SBATCH --gpus=4
#SBATCH --mem=250GB

cd commands
srun ./run_train.sh
