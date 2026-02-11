import numpy as np
import torch
from typing import Optional, Union

from omegaconf import OmegaConf, ListConfig, DictConfig
from tqdm import tqdm

from unet3d2d import SimAQModel
from datagenerator import quantile_normalize_tensor


class Simaq:
    """SimAQ model wrapper for inference."""
    
    def __init__(self, config: Union[str, DictConfig, ListConfig], weights: str, device: Optional[str] = None):
        """
        Initialize SimAQ model.
        
        Args:
            config: Path to config file or OmegaConf config object
            weights: Path to model weights file
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        # Load config if it's a path
        if isinstance(config, str):
            self.cfg = OmegaConf.load(config)
            OmegaConf.resolve(self.cfg)
        else:
            self.cfg = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = SimAQModel(
            in_channels=self.cfg.model.in_channels,
            out_channels=self.cfg.model.out_channels,
            init_features=self.cfg.model.init_features,
            depth=self.cfg.model.depth,
            skip_first=self.cfg.model.skip_first
        ).to(self.device)
        
        # Load weights
        print(f"Loading model weights from {weights}")
        weights_dict = torch.load(weights, map_location='cpu')
        self.model.load_state_dict(weights_dict, strict=True)
        self.model.eval()
        
        self.padding = self.cfg.data.padding
    
    def predict(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Perform inference on a 3D image.
        
        Args:
            img: Input 3D image as numpy array (D, H, W)
            normalize: Whether to apply quantile normalization
            
        Returns:
            Segmentation prediction as numpy array (D, H, W) with dtype uint8
        """
        # Convert to tensor
        in_tensor = torch.from_numpy(img).float()
        
        # Normalize if requested
        if normalize:
            in_tensor = quantile_normalize_tensor(in_tensor)
        
        # Prepare output tensor
        out_tensor = torch.zeros_like(in_tensor)
        
        # Inference
        with torch.no_grad():
            for i in tqdm(range(self.padding, in_tensor.shape[0] - self.padding)):
                img_slice = in_tensor[i - self.padding:i + self.padding + 1, :, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
                img_slice = img_slice.to(self.device)
                pred_slice = self.model(img_slice)
                out_tensor[i, :, :] = torch.argmax(pred_slice, dim=1).squeeze().cpu()
        
        return out_tensor.cpu().numpy().astype(np.uint8)