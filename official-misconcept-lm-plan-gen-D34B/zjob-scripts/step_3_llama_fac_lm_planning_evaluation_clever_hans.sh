#!/bin/bash


export PYTHONPATH=${PWD}
export WORKING_DIR=${PWD}

export HF_HOME=${PWD}/data/01_raw/huggingface   

export CUDA_HOME=$CONDA_PREFIX
export TRITON_CACHE_DIR=${PWD}/data/02_intermediate/triton_cache
export TOKENIZERS_PARALLELISM=false
    
config_type=$1


export EVAL_CONFIG_TYPE=$config_type

# check cuda_visible_devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
the_num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


# NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL works for A100 GPUs
# ! set three different TOP_K_CONFIG values
ACTION_HINT_SIZE=15 TOP_K_CONFIG=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL accelerate launch --num_processes $the_num_processes src/better_language_model_for_plan_generation/pipelines/language_model_planning_evaluation/nodes.py




