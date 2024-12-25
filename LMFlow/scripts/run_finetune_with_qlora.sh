#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
# model_name_or_path=meta-llama/Llama-2-13b-hf
# dataset_path=data/alpaca/train_conversation
# conversation_template=llama2
# output_dir=output_models/finetune
# deepspeed_args="--master_port=11000"



export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_API_KEY=b23f406cf873dad9d573cdc6868e8ee14fa1a0db

model_name_or_path=/data/data_public/breeze/models/Qwen/Qwen2-7B
# model_name_or_path=/data/data_public/ysq/models/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
#RAIT数据集
# dataset_path=/data/data_public/breeze/KnowledgeBoundary/2.2.2_TASK/2.2.2_0_ModelGenData/Qwen2-7B

dataset_path=/data/data_public/breeze/KnowledgeBoundary/2.2.2_TASK/2.2.2_0_ModelGenData/models--Qwen--Qwen2.5-3B
# #筛选后的数据集
# dataset_path=/data/data_public/breeze/KnowledgeBoundary/LMFlow/2.2.2_2_RES_DATASET/llama3b

output_dir=/data/data_public/breeze/output_models/finetuned_QWen2_7b_MMLU_N
#output_dir=/data/data_public/breeze/output_models/finetuned_QWen2.5_3b_MMLU_N
deepspeed_args="--master_port=11000"
conversation_template=qwen2

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune_with_lora
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --conversation_template ${conversation_template} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.01 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --use_qlora 1 \
    --save_aggregated_lora 0 \
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err