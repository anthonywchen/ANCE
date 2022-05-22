#!/bin/bash
#
#SBATCH --job-name=train_bm25
#SBATCH --time=48:00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=6
#SBATCH --mem=200GB

srun python -m torch.distributed.launch --nproc_per_node=8 drivers/run_warmup.py \
   --train_model_type rdot_nll \
   --model_name_or_path roberta-base \
   --task_name MSMarco \
   --do_train \
   --data_dir data/raw_data/ \
   --max_seq_length 128 \
   --evaluate_during_training \
   --per_gpu_eval_batch_size 256 \
   --per_gpu_train_batch_size 32 \
   --learning_rate 5e-5  \
   --logging_steps 50000 \
   --num_train_epochs 1  \
   --output_dir outputs/bm25 \
   --warmup_steps 1000  \
   --overwrite_output_dir \
   --save_steps 50000 \
   --gradient_accumulation_steps 1 \
   --expected_train_size 35000000 \
   --logging_steps_per_eval 1 \
   --fp16 \
   --optimizer lamb \
   --log_dir tensorboard/bm25/logs