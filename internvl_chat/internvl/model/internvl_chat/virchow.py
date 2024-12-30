from huggingface_hub import login
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
from transformers import AutoConfig

# Needs to be updated
login("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")


# need to specify MLP layer and activation function for proper init
def get_virchow_model(task="eval"):
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    if task == "eval":
        model.eval()
    model.to("cuda")
    return model


##
def compress_3d_tensor_mean_torch(tensor, target_dim=1024):
    batch_size, seq_len, original_dim = tensor.shape
    factor = original_dim // target_dim

    # Reshape the last dimension from 1280 to (1024, factor) and compute the mean over the last dimension
    compressed_tensor = (
        tensor[:, :, : target_dim * factor]
        .view(batch_size, seq_len, target_dim, factor)
        .mean(dim=-1)
    )

    return compressed_tensor
