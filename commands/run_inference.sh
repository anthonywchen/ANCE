# # Passage ANCE(FirstP) 
gpu_no=8
seq_length=128
batch_size=256
model_type=rdot_nll
tokenizer_type="roberta-base"
base_data_dir="../data/raw_data/"
data_dir="../data/processed_data/"
job_name="bm25"
pretrained_checkpoint_dir="../outputs/bm25"

model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data_inf/"

cmd=" \
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py \
  --training_dir $pretrained_checkpoint_dir \
  --init_model_dir $pretrained_checkpoint_dir \
  --model_type $model_type \
  --output_dir $model_ann_data_dir \
  --cache_dir "${model_ann_data_dir}cache/" \
  --data_dir $data_dir \
  --max_seq_length $seq_length \
  --per_gpu_eval_batch_size $batch_size \
  --topk_training 200
  --negative_sample 20 \
  --end_output_num 0 \
  --inference
"
echo $cmd
eval $cmd
