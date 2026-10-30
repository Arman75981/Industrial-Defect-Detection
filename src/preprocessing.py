"""
Preprocessing module for Industrial Defect Detection.
Ensures identical preprocessing during training, evaluation, and real-time inference.
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard transformation pipeline
def get_transforms(image_size: Tuple[int, int] = (128, 128)) -> T.Compose:
    """Returns standard torchvision transforms pipeline."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),  # Converts PIL Image [0, 255] to Tensor [0.0, 1.0] with shape (C, H, W)
    ])

def preprocess_image(image_input: Union[str, Path, Image.Image, np.ndarray], 
                     image_size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Loads and preprocesses a single image.
    
    Args:
        image_input: File path, PIL Image, or NumPy array.
        image_size: Desired (height, width).
        
    Returns:
        Tensor of shape (3, H, W) with pixel values in range [0.0, 1.0].
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
        
    transform = get_transforms(image_size)
    tensor = transform(img)
    return tensor

def deprocess_image(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a normalized tensor [0.0, 1.0] back to an RGB uint8 image [0, 255]
    for visualization and OpenCV / PIL operations.
    
    Args:
        tensor: Tensor of shape (3, H, W) or (1, 3, H, W), or NumPy array.
        
    Returns:
        NumPy array of shape (H, W, 3) with dtype uint8.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)  # Remove batch dim
        # (C, H, W) -> (H, W, C)
        array = tensor.permute(1, 2, 0).numpy()
    else:
        array = np.array(tensor)
        if array.ndim == 4:
            array = array[0]
        if array.shape[0] == 3:  # (3, H, W)
            array = np.transpose(array, (1, 2, 0))

    # Clip to [0, 1] range to avoid display artifacts
    array = np.clip(array, 0.0, 1.0)
    uint8_img = (array * 255.0).round().astype(np.uint8)
    return uint8_img

def load_mask(mask_path: Union[str, Path], 
              image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Loads and binarizes a ground-truth defect mask.
    
    Args:
        mask_path: Path to the mask image file.
        image_size: Desired (height, width).
        
    Returns:
        Binary NumPy array of shape (H, W) with values 0 (normal) or 1 (defect).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize(image_size, Image.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.float32)
    binary_mask = (mask_arr > 127.5).astype(np.uint8)
    return binary_mask
