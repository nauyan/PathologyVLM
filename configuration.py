from transformers import AutoModel, AutoTokenizer
import torch
# Load the model and tokenizer
model_path = './internvl_chat/pretrained/InternVL2-1B'
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Get the model configuration
model_config = model.config
print(model_config)
