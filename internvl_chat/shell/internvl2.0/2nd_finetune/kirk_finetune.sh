#!/bin/bash

# Set your GPU and batch size variables
GPUS=1
PER_DEVICE_BATCH_SIZE=1
NUM_EPOCHS=5

# Run the original script with hardcoded USE_BACKBONE_LORA and USE_LLM_LORA values

# First run: USE_BACKBONE_LORA=0, USE_LLM_LORA=8
echo "Running fine-tuning with USE_BACKBONE_LORA=0 and USE_LLM_LORA=8"
OUTPUT_DIR="work_dirs/custom_kirc_use_backbone_lora_0_use_llm_lora_8_mlp_epoch_${NUM_EPOCHS}"
mkdir -p "$OUTPUT_DIR"
GPUS=${GPUS} NUM_EPOCHS=${NUM_EPOCHS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} USE_BACKBONE_LORA=0 USE_LLM_LORA=8 OUTPUT_DIR=${OUTPUT_DIR} sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh

# Second run: USE_BACKBONE_LORA=8, USE_LLM_LORA=8
echo "Running fine-tuning with USE_BACKBONE_LORA=8 and USE_LLM_LORA=8"
OUTPUT_DIR="work_dirs/custom_kirc_use_backbone_lora_8_use_llm_lora_8_mlp_epoch_${NUM_EPOCHS}"
mkdir -p "$OUTPUT_DIR"
GPUS=${GPUS} NUM_EPOCHS=${NUM_EPOCHS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} USE_BACKBONE_LORA=8 USE_LLM_LORA=8 OUTPUT_DIR=${OUTPUT_DIR} sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh

# Third run: USE_BACKBONE_LORA=16, USE_LLM_LORA=16
echo "Running fine-tuning with USE_BACKBONE_LORA=16 and USE_LLM_LORA=16"
OUTPUT_DIR="work_dirs/custom_kirc_use_backbone_lora_16_use_llm_lora_16_mlp_epoch_${NUM_EPOCHS}"
mkdir -p "$OUTPUT_DIR"
GPUS=${GPUS} NUM_EPOCHS=${NUM_EPOCHS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} USE_BACKBONE_LORA=16 USE_LLM_LORA=16 OUTPUT_DIR=${OUTPUT_DIR} sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
