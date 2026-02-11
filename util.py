import torch
from random import random
from typing import Optional
from torch.optim.swa_utils import AveragedModel


def single_to_multi_channel(image: torch.Tensor, num_classes: Optional[int]=None) -> torch.Tensor:
    """
    Convert an image tensor with discrete class labels to a multi-channel tensor.
    0 is assumed to be the background class, and each subsequent integer represents a different class.
    Each channel corresponds to a class, with values 0 or 1 indicating the presence of that class.
    Args:
        image (torch.Tensor): Input tensor of shape (B, H, W) with class labels.
    Returns:
        torch.Tensor: Multi-channel tensor of shape (B, C, H, W) where C is the number of classes.
    """
    if image.dim() != 3:
        raise ValueError("Input tensor must be of shape (B, H, W)")

    batch_size, height, width = image.shape
    if not num_classes:
        num_classes = int(image.max().item()) + 1  # Assuming classes are labeled from 0 to C-1

    multi_channel_image = torch.zeros((batch_size, num_classes, height, width), dtype=torch.float32)

    for c in range(num_classes):
        multi_channel_image[:, c, :, :] = (image == c+1).float() # + 1 to shift class labels from 0 to 1, as 0 is background

    return multi_channel_image


class EMAWrapper(AveragedModel):
    def __init__(self, model, decay):
        super().__init__(model)
        self.decay = decay

    # Create a property 'device' that refers to self.module.device
    @property
    def device(self):
        return self.module.device

    def update_parameters(self, model):
        # Use EMA instead of SWA-style average
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            if p_model.requires_grad:
                p_swa.data.mul_(self.decay).add_(p_model.data, alpha=1 - self.decay)

def chance(p: float) -> bool:
    return random() < p

def clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))