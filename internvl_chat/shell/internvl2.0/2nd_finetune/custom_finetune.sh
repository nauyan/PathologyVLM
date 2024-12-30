#!/bin/bash

# Set your GPU and batch size variables
GPUS=1
PER_DEVICE_BATCH_SIZE=1
NUM_EPOCHS=5

# Define an array of use_lora values


# Loop over each use_lora value
for use_lora in 8; do
  echo "Running fine-tuning with use_lora=${use_lora}"

  # Set OUTPUT_DIR dynamically based on use_lora value
  OUTPUT_DIR="work_dirs/custom_tcga_512_use_backbone_lora_${use_lora}_use_llm_lora_${use_lora}_mlp_epoch_${NUM_EPOCHS}"

  # Create the directory if it doesn't exist
  if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
  fi

  # Run the original script with the updated use_lora value and OUTPUT_DIR
  GPUS=${GPUS} NUM_EPOCHS=${NUM_EPOCHS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} USE_BACKBONE_LORA=${use_lora} USE_LLM_LORA=${use_lora} OUTPUT_DIR=${OUTPUT_DIR} sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
done
