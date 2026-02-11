import torch
from typing import Dict, Any, Tuple
from omegaconf import ListConfig, DictConfig

import torch.nn as nn
import torchvision.transforms.functional as TF

from util import chance
from datagenerator import generate_elastic_displacement

class Augmentor(nn.Module):
    """
    Augmentor module that applies various augmentations to images and masks.
    Images are expected to have shape (B, C, D, H, W) where D is the depth governed by the padding,
    while masks have the shape (B, C, H, W).
    
    Args:
        cfg: Configuration dictionary specifying augmentation parameters
        **kwargs: Additional arguments to override cfg values
    """
    
    def __init__(self, cfg: ListConfig | DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg.copy()
        self.cfg.update(kwargs)
        
        # Extract augmentation settings
        self.flip_prob = self.cfg.augmentations.flip_prob

        self.elastic_deform_prob = self.cfg.augmentations.elastic_deform_prob
        self.elastic_deform_alpha = self.cfg.augmentations.elastic_deform_alpha
        self.elastic_deform_sigma = self.cfg.augmentations.elastic_deform_sigma

        self.resized_crop_prob = self.cfg.augmentations.resized_crop_prob
        self.resized_crop_scale_min = self.cfg.augmentations.resized_crop_scale_min

        self.brightness_prob = self.cfg.augmentations.brightness_prob
        self.brightness_factor_max = self.cfg.augmentations.brightness_factor_max
        self.contrast_prob = self.cfg.augmentations.contrast_prob
        self.contrast_factor_max = self.cfg.augmentations.contrast_factor_max

        self.noise_prob = self.cfg.augmentations.noise_prob
        self.noise_std_max = self.cfg.augmentations.noise_std_max

        self.blur_prob = self.cfg.augmentations.blur_prob

        self.invert_prob = self.cfg.augmentations.invert_prob
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to the input image and mask.
        
        :param image: Input image tensor (B, C, D, H, W)
        :type image: torch.Tensor
        :param mask: Input mask tensor (B, C, D, H, W)
        :type mask: torch.Tensor
        :return: Augmented image and mask tensors
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, _, D, H, W = image.shape
        
        # Vertical Flip
        if chance(self.flip_prob):
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # Horizontal Flip
        if chance(self.flip_prob):
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Elastic Deformation
        if chance(self.elastic_deform_prob):
            displacement = generate_elastic_displacement(
                alpha=self.elastic_deform_alpha,
                sigma=self.elastic_deform_sigma,
                size = [H, W],
                device=image.device
            )
            for d in range(D):
                image[:,0,d] = TF.elastic_transform(image[:,0,d], displacement, 
                                         interpolation=TF.InterpolationMode.BICUBIC, 
                                         fill=0.0) # type: ignore
            # Mask has 1 channel and no depth dimension
            mask[:,0] = TF.elastic_transform(mask[:,0], displacement, 
                                    interpolation=TF.InterpolationMode.NEAREST, 
                                    fill=-1) # type: ignore
        # Resized Crop
        if chance(self.resized_crop_prob):
            scale = torch.empty(1).uniform_(self.resized_crop_scale_min, 1.0).item()
            new_h = int(H * scale)
            new_w = int(W * scale)
            left = torch.randint(0, H - new_h + 1, (1,)).item()
            front = torch.randint(0, W - new_w + 1, (1,)).item()
            image = image[:, :, :, left:left+new_h, front:front+new_w]
            image = torch.nn.functional.interpolate(image, size=(D, H, W), mode='area')
            mask = mask[:, :, left:left+new_h, front:front+new_w]
            mask = torch.nn.functional.interpolate(mask.float(), size=(H, W), mode='nearest').long()

        # Brightness Adjustment
        if chance(self.brightness_prob):
            factor = torch.empty(1).uniform_(-self.brightness_factor_max, self.brightness_factor_max).item()
            image = image + image.max() * factor
        
        # Contrast Adjustment
        if chance(self.contrast_prob):
            factor = torch.empty(1).uniform_(1.0 - self.contrast_factor_max, 1.0 + self.contrast_factor_max).item()
            image = (image - image.mean()) * factor + image.mean()

        # Gaussian Blur
        if chance(self.blur_prob):  
            sigma = torch.empty(1).uniform_(0.1, 2.0).item()
            kernel_size = [3, 3]
            for d in range(D):
                image[:,0,d] = TF.gaussian_blur(image[:,0,d], kernel_size=kernel_size, sigma=[sigma, sigma])

        # Gaussian Noise
        if chance(self.noise_prob):
            std = torch.empty(1).uniform_(0.0, self.noise_std_max).item()
            noise = torch.randn_like(image) * std
            image = image + noise

        # Inversion
        if chance(self.invert_prob):
            image = -image

        return image, mask