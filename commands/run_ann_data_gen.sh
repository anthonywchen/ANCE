#!/bin/bash
#
# This script is for generate ann data for a model in training
#
# For the overall design of the ann driver, check run_train.sh
#
# This script continuously generate ann data using latest model from model_dir
# For training, run this script after initial ann data is created from run_train.sh
# Make sure parameter used here is consistent with the training script

#  Passage ANCE(FirstP)
gpu_no=4
seq_length=128
model_type=rdot_nll
tokenizer_type="roberta-base"
base_data_dir="../data/raw_data/"
preprocessed_data_dir="../data/processed_data/"
negative_sample=4
topk_training=200
ann_chunk_factor=5
job_name="ance_train_${topk_training}_${negative_sample}_${ann_chunk_factor}"

##################################### Inital ANN Data generation ################################
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"

initial_data_gen_cmd="\
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py \
  --training_dir $model_dir \
  --model_type $model_type \
  --output_dir $model_ann_data_dir \
  --cache_dir "${model_ann_data_dir}cache/" \
  --data_dir $preprocessed_data_dir \
  --max_seq_length $seq_length \
  --ann_chunk_factor $ann_chunk_factor \
  --per_gpu_eval_batch_size 256 \
  --topk_training $topk_training \
  --negative_sample $negative_sample \
  --fp16 \
  --approx_search
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd
