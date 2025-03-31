#!/bin/bash



if [[ $(hostname) == *"spartan"* ]]; then
    echo "Running on Spartan"  
else
    echo "Running on local"
fi

export PYTHONPATH=${PWD}
export WORKING_DIR=${PWD}
# check if hostname contain spartan, change HF_HOME to another location 

export HF_HOME=${PWD}/data/01_raw/huggingface   



export CUDA_HOME=$CONDA_PREFIX
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_CACHE_DIR=${PWD}/data/02_intermediate/triton_cache

config_type=$1

llamafactory-cli train zjob-scripts/llama_fac_config/deepspeed/llama3_full_sft_ds3_${config_type}.yaml
