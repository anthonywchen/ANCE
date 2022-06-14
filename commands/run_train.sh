#!/bin/bash
#
# This script is for training with updated ann driver
#
# The design for this ann driver is to have 2 separate processes for training: one for passage/query 
# inference using trained checkpoint to generate ann data and calculate ndcg, another for training the model
# using the ann data generated. Data between processes is shared on common directory, model_dir for checkpoints
# and model_ann_data_dir for ann data.
#
# This script initialize the training and start the model training process
# It first preprocess the msmarco data into indexable cache, then generate a single initial ann data
# version to train on, after which it start training on the generated ann data, continously looking for
# newest ann data generated in model_ann_data_dir
#
# To start training, you'll need to run this script first
# after intial ann data is created (you can tell by either finding "successfully created 
# initial ann training data" in console output or if you start seeing new model on tensorboard),
# start run_ann_data_gen.sh in another dlts job (or same dlts job using split GPU)
#
# Note if preprocess directory or ann data directory already exist, those steps will be skipped
# and training will start immediately

 # Passage ANCE (FirstP)
data_type=1
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
pretrained_checkpoint_dir="../outputs/bm25/checkpoint-150000"

##################################### Data Preprocessing ################################
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"

preprocess_cmd="\
python ../data/msmarco_data.py \
  --data_dir $base_data_dir \
  --out_data_dir $preprocessed_data_dir \
  --model_type $model_type \
  --model_name_or_path $tokenizer_type \
  --max_seq_length $seq_length \
  --data_type $data_type \
"
#echo -e $preprocess_cmd "\n"
#eval $preprocess_cmd

######## Initial ANN Data generation ########
initial_data_gen_cmd="\
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py \
  --training_dir $model_dir \
  --init_model_dir $pretrained_checkpoint_dir \
  --model_type $model_type \
  --output_dir $model_ann_data_dir \
  --cache_dir "${model_ann_data_dir}cache/" \
  --data_dir $preprocessed_data_dir \
  --max_seq_length $seq_length \
  --ann_chunk_factor $ann_chunk_factor \
  --per_gpu_eval_batch_size 256 \
  --topk_training $topk_training \
  --negative_sample $negative_sample \
  --end_output_num 0 \
  --fp16 \
  --approx_search
"
echo $initial_data_gen_cmd
eval $initial_data_gen_cmd

######## Training ########
warmup_steps=5000
per_gpu_train_batch_size=16
gradient_accumulation_steps=1
learning_rate=5e-6
max_steps=300000
logging_steps=500
save_steps=10000

train_cmd="\
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann.py \
  --model_type $model_type \
  --model_name_or_path $pretrained_checkpoint_dir \
  --task_name MSMarco \
  --triplet \
  --data_dir $preprocessed_data_dir \
  --ann_dir $model_ann_data_dir \
  --max_seq_length $seq_length \
  --per_gpu_train_batch_size=$per_gpu_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --learning_rate $learning_rate \
  --output_dir $model_dir \
  --max_steps $max_steps \
  --warmup_steps $warmup_steps \
  --logging_steps $logging_steps \
  --save_steps $save_steps \
  --optimizer lamb \
  --fp16
  --single_warmup \
  --log_dir ../tensorboard/$job_name
"

#echo $train_cmd
#eval $train_cmd
