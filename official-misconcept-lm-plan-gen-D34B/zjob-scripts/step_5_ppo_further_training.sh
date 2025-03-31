#!/bin/bash



export PYTHONPATH=${PWD}
export WORKING_DIR=${PWD}
# check if hostname contain spartan, change HF_HOME to another location 

export HF_HOME=${PWD}/data/01_raw/huggingface   

export CUDA_HOME=$CONDA_PREFIX
export TRITON_CACHE_DIR=${PWD}/data/02_intermediate/triton_cache
export TOKENIZERS_PARALLELISM=false
export DEEPSPEED_TIMEOUT=1800000
    
config_type=$1

export EVAL_CONFIG_TYPE=$config_type




# check cuda_visible_devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0 # DEBUG
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
the_num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
accelerate launch --num_processes $the_num_processes --config_file zjob-scripts/llama_fac_config/deepspeed/deepspeed_zero2.yaml src/better_language_model_for_plan_generation/pipelines/further_ppo_training/nodes.py --output_dir data/07_model_output/qwen/full_t0/ppo

# NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL works for A100 GPUs
