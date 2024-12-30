## Note:
This project utilizes code from internvl github repository that comprises of vision language model to perform training on various task and also provide script to perform evaluation.


# Introduction
THe motivation of this project is to evaluate the capability of vision language model in the domain of Computation Pathology for generating report that involves sruvival-analysis and tumor grade classification. We build a custom vision language model by integrating a vision encoder virchow-2 by paige-ai trained on large corpus of histopathological images. This custom vision encoder allows to extract enrich features from WSI-patches that are further transfered to large language model to generate the report.Some of the important script in this proejct are as follows:

- `internvl_chat/internvl/train/internvl_chat_finetune.py:` This script is responsible for training the vlm
- `internvl_chat/internvl/model/internvl_chat/virchow.py:` This script is responsible for authenticating and returning the virchow-2 vision encoder
- `internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py:` THis script is responsible for designing the complete architecture of the vision language model which involves vision-encoder, projection-layer, and finally the large language model.
- `internvl_chat/internvl/inference_v2.py:` This scirpt is written for inference more of the details in further section
- `internvl_chat/internvl/custom_eval.py:` THis script is responsible for evaluating vision language model.

# Setup
## Download Pre-trained weights:
First navigate to pretrained directory by `cd pretrained` and then run the following command to install pre-trained weights:
`huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B`

## Setting Up Data:
First create/navigate to `internvl/data` directory and then place your dataset, an example of this is:
```bash
data/coco
├── annotations
│   ├── coco_karpathy_test.json
│   ├── coco_karpathy_test_gt.json
│   └── coco_karpathy_train_567k.jsonl
├── train2014
├── val2014
└── test2015
```

After that create a json file in the directory `internvl_chat/shell/data/coco_caption.json` with the following format:
```bash
{
  "coco_karpathy_train_567k": {
    "root": "data/coco/",
    "annotation": "data/coco/annotations/coco_karpathy_train_567k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 566747
  }
}
```

Finally set the `--metapath` flag in the bash script `internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh` to `./shell/data/{dataset_name}`

# Training
# Vision-Language Model Training

After completing the setup, you can fine-tune the Vision-Language Model using the following training script. Below are the customizable flags used in the training process and their respective explanations.

---

### Training Flags and Explanations

- `--use_llm_lora`  
  Determines whether to apply Low-Rank Adaptation (LoRA) for the Language Model (LLM). LoRA optimizes fine-tuning by updating a small number of parameters, which helps in reducing memory usage and computational cost.

- `--use_backbone_lora`  
  Enables LoRA for the vision encoder (backbone). This is particularly useful for optimizing storage and computational efficiency during fine-tuning.

- `--freeze_llm`  
  Specifies whether to freeze the language model's parameters.  
  - Example: `--freeze_llm True` will freeze the LLM and prevent it from being updated during training.  
  - Default: `False`.

- `--freeze_mlp`  
  Controls whether to freeze the Multi-Layer Perceptron (MLP) layers in the model.  
  - Example: `--freeze_mlp False` allows fine-tuning of MLP layers.  
  - Default: `True`.

- `--freeze_backbone`  
  Indicates whether to freeze the vision encoder (backbone).  
  - Example: `--freeze_backbone True` will freeze the vision encoder, preventing its parameters from being updated during training.  
  - Default: `False`.

---

### Additional Parameters

- `GPUS`  
  Defines the number of GPUs to be used for training.  
  - Example: `GPUS=8`.

- `PER_DEVICE_BATCH_SIZE`  
  Defines the batch size per GPU.  
  - Example: `PER_DEVICE_BATCH_SIZE=4`.

---

### Example Command

Here is an example of how you can use the above flags in a training script:

```bash
GPUS=8 PER_DEVICE_BATCH_SIZE=4 sh shell/internvl2.0/2nd_finetune/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh \
  --use_llm_lora 8\
  --use_backbone_lora 8\
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True
