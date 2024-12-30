import torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer
import logging
import json
import argparse
from internvl.model.internvl_chat.virchow import get_virchow_model
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="InternVL Image Captioning")
    parser.add_argument("--model_path", type=str, default="pretrained/InternVL2_1B", help="Path to the pretrained model")
    parser.add_argument("--checkpoint_path", type=str, default='work_dirs/custom_tcga_512_use_backbone_lora_8_use_llm_lora_8_mlp_epoch_5/checkpoint-52400/global_step52400/mp_rank_00_model_states.pt', help="Path to the fine-tuned checkpoint")
    parser.add_argument("--image_path", type=str, default='data/TCGA_GBM_LLG_512/test/TCGA-02-0003-01Z-00-DX3.995C2924-E298-4517-82A4-15806766CE31_1_512_0.png', help="Path to the input image")
    parser.add_argument("--image_path_train", type=str, default='data/TCGA_GBM_LLG_512/train/TCGA-VM-A8CA-01Z-00-DX6.07221643-D7F3-4E38-9A63-C7B3B46E570B_1_512_512.png', help="Path to the input image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--use_llm_lora", type=int, default=8, help="Adapter rank on LLM"),
    parser.add_argument('--use_backbone_lora',type=int, default=8, help='Adpater rank for vision model'),
    return parser.parse_args()

def build_transform(input_size=224):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_path, input_size=224):
    try:
        image = Image.open(image_path).convert("RGB")
        transform = build_transform(input_size)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise

def generate_caption(model, tokenizer, image_tensor):
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    question = '<image>\n# Histopathology Image Analysis\n\nAnalyze the provided histopathology image and generate a structured report with the following information:\n\n1. Classification Label: [Provide a numerical value between 0.0 and 1.0]\n2. Survival Time: [Provide an estimate in months, rounded to one decimal place]\n3. Censored: [Provide either 0.0 for uncensored or 1.0 for censored]\n\nPlease ensure all three values are always provided, even if you\'re uncertain. Use your best judgment based on the image characteristics. Format your response as follows:\n\nClassification Label: [value], Survival Time: [value] months, Censored: [value]\n\nExample response:\nClassification Label: 0.7, Survival Time: 48.5 months, Censored: 1.0\n\nImportant: Always provide numerical values for all three categories. Do not leave any field blank or use text descriptions instead of numbers.'
    try:
        response = model.chat(tokenizer, image_tensor, question, generation_config)
        return response
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        raise

def load_model_and_tokenizer(args):
    try:
        config_path = f"{args.model_path}/config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config = InternVLChatConfig(**config_data)
        logger.info('Loaded config')

        # Configure model settings
        config.vision_config.drop_path_rate = 0.0
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for LLaMA')

        config.template = 'Hermes-2'
        config.select_layer = -1
        config.dynamic_image_size = True
        config.use_thumbnail = True
        config.ps_version = 'v2'
        config.min_dynamic_patch = 1
        config.max_dynamic_patch = 6
        config.force_image_size = args.input_size
        # config.use_llm_lora = args.use_llm_lora
        # config.use_backbone_lora = args.use_backbone_lora

        vision_model = get_virchow_model()
        # logger.info(f'Vision model config default: {vision_model.default_cfg}')
        # logger.info(f'Default vision configuration: {config.vision_config}')

        model = InternVLChatModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, config=config, vision_model=vision_model)

        if args.use_backbone_lora:
            logger.info('applying backbone lora')
            logger.info(f'using back bone lora')
            model.wrap_backbone_lora(r=args.use_backbone_lora, lora_alpha=2 * args.use_backbone_lora)
            model.config.use_backbone_lora = args.use_backbone_lora

        if args.use_llm_lora:
            logger.info(f'using llm lora')
            model.wrap_llm_lora(r=args.use_llm_lora, lora_alpha=2 * args.use_llm_lora)
            model.config.use_llm_lora = args.use_llm_lora


        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        logger.info('Loaded checkpoint')
        logger.info(f'Checkpoint keys: {checkpoint.keys()}')
        model.load_state_dict(checkpoint['module'])
        model.to(args.device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def main():
    args = parse_args()
    
    try:
        model, tokenizer = load_model_and_tokenizer(args)
        
        image_tensor = load_image(args.image_path, args.input_size).to(args.device)
        caption = generate_caption(model, tokenizer, image_tensor)
        print('\n\nActual Value is: <Classification label>: 2.0, <Survival time>: 144.0 months, <Censored>: 0.0')
        print("Test Generated Caption:", caption)

        image_tensor = load_image(args.image_path_train, args.input_size).to(args.device)
        caption = generate_caption(model, tokenizer, image_tensor)
        print('\n\n Actual Value is: <Classification label>: 0.0, <Survival time>: 411.0 months, <Censored>: 1.0..')
        print('Train generated caption :',caption)

        
    except Exception as e:
        logger.error(f"An error occurred during inference: {str(e)}")

if __name__ == "__main__":
    main()