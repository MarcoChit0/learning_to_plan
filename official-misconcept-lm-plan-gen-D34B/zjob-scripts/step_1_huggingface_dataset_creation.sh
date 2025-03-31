#!/bin/bash


export PYTHONPATH=${PWD}
export WORKING_DIR=${PWD}
export HF_HOME=${PWD}/data/01_raw/huggingface
export CUDA_HOME=$CONDA_PREFIX
export TRITON_CACHE_DIR=${PWD}/data/02_intermediate/triton_cache

kedro run -p text_prompt_generation     