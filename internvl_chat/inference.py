import torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel, AutoTokenizer
import logging
from internvl.model.internvl_chat.virchow import get_virchow_model
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
# from internvl.model.internvl_chat import InternVLChatConfig, InternVisionConfig

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and attach it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


        # config.vision_config.drop_path_rate = model_args.drop_path_rate
        # if config.llm_config.model_type == 'internlm2':
        #     config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        #     logger.info('Using flash_attention_2 for InternLM')
        # else:
        #     config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        #     logger.info('Using flash_attention_2 for LLaMA')
        # config.template = data_args.conv_style
        # config.select_layer = model_args.vision_select_layer
        # config.dynamic_image_size = data_args.dynamic_image_size
        # config.use_thumbnail = data_args.use_thumbnail
        # config.ps_version = model_args.ps_version
        # config.min_dynamic_patch = data_args.min_dynamic_patch
        # config.max_dynamic_patch = data_args.max_dynamic_patch
        # logger.info(f'Configuration of INternvl is : {config}')



# Define the transformation for the image
def build_transform(input_size=224):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

# Load and preprocess the image
def load_image(image_path, input_size=224):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to generate a caption
def generate_caption(model, tokenizer, image_tensor):
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # Prepare the input with the token for image prompt
    question = '<image>\nGenerate report for the following histopathology image.'
    # pixel_values = image_tensor.to(torch.bfloat16).cuda()
    pixel_values = image_tensor.cuda()

    # Generate the caption
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

# Load your model and tokenizer from the specified path
model_path = 'pretrained/InternVL2_1B'

import json
# Load the configuration from a JSON file
config_path = model_path + '/config.json'

with open(config_path, 'r') as f:
    config_data = json.load(f)

# Assuming the vision_config is part of the config JSON, extract it as a dictionary
vision_config = config_data.get('vision_config', {})

config = InternVLChatConfig(**config_data)
logger.info('loaded config')
config.vision_config.drop_path_rate = 0.0
if config.llm_config.model_type == 'internlm2':
    config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
    logger.info('Using flash_attention_2 for InternLM')
else:
    config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
    logger.info('Using flash_attention_2 for LLaMA')
config.template = 'Hermes-2'
config.select_layer = -1
config.dynamic_image_size = True
config.use_thumbnail = True
config.ps_version = 'v2'
config.min_dynamic_patch = 1
config.max_dynamic_patch = 6
config.force_image_size = 224
# config.vision_config.hidden_size= 1280
config.use_llm_lora=16

vision_model = get_virchow_model()
logger.info(f'vision model config default: {vision_model.default_cfg}')
logger.info(f'default vision configuration: {config.vision_config}')
# logger.info(f'vision config: {config.vision_config}')
model = InternVLChatModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, config=config, vision_model=vision_model)
# checkpoint = torch.load('work_dirs/lora_0_mlp_epoch_15/checkpoint-44000/global_step44000/mp_rank_00_model_states.pt')
# checkpoint = torch.load('./work_dirs/lora_0_mlp_epoch_15/checkpoint-44000/global_step44000/mp_rank_00_model_states.pt')
# checkpoint = torch.load('./lora_epochs_10/checkpoint-29400/global_step29400/mp_rank_00_model_states.pt')
checkpoint = torch.load('work_dirs/lora_epochs_25/checkpoint-73400/global_step73400/mp_rank_00_model_states.pt')
logger.info(f'loaded checkpoint')
logger.info(f'checkpoint keys: {checkpoint.keys()}')
model.load_state_dict(checkpoint['module'])
# logger.info(f'Configuraiton is: {config}')

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Path to the image you want to caption
image_path = './data/pubmed_dataset/train/0a0a7c19-862b-441e-afa7-76295542c576.jpg'
model.to('cuda')
# Load the image and generate the caption
image_tensor = load_image(image_path)
caption = generate_caption(model, tokenizer, image_tensor)

# Print the generated caption
print("Generated Caption:", caption)
