#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_API_KEY=b23f406cf873dad9d573cdc6868e8ee14fa1a0db

model_name_or_path=/data/data_public/breeze/models/openlm-research/open_llama_3b

# #RAIT数据集
# dataset_path=/data/data_public/breeze/KnowledgeBoundary/LMFlow/2.2.2_0_ModelGenData/open_llama_3b

#筛选后的数据集
dataset_path=/data/data_public/breeze/KnowledgeBoundary/LMFlow/2.2.2_2_RES_DATASET/llama3b

output_dir=/data/data_public/breeze//output_models/finetuned_2_llama_3b 

deepspeed_args="--master_port=11000"

conversation_template=llama2

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
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
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
exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --num_train_epochs 0.01 \
    --learning_rate 2e-5 \
    --disable_group_texts 1 \
    --block_size 256 \
    --per_device_train_batch_size 1 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
